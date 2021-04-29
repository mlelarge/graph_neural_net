import json
import copy
import os

from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
from torch import sigmoid as Sigmoid
from torch.nn import Softmax
from toolbox.logger import Experiment
import loaders.data_generator as dg
import toolbox.metrics as metrics
import toolbox.losses as losses
import toolbox.utils as utils

from abc import abstractmethod


#This file is here to create the handler for each experiment.
#
#To create a new experiment, you need to:
# 1) create an 'Experiment' class which inherits from the 'Experiment_Helper'
# 2) in its '__init__' function, you need to define:
#       - self.generator : the generator of the dataset
#       - self.criterion : the criterion/loss function/class (can be seen in toolbox/losses.py for my functions)
#       - self.eval_function and self.metric : another function of seeing progress of learning. For now are only implemented:
#               ¤ loss : if the method is a loss of some sort. The function needs to return only a scalar
#               ¤ acc  : for accuracy. The function needs to return (true_positives, total_sample) => acc = true_positives/total_sample
#               ¤ f1   : for an F1-Score. The function needs to return a tuple (precision, recall, f1_score)
#                   (To implement other metrics, go into the 'Experiment_Helper.init_update_eval' function)
#       - Don't forget to call the superclass initialization AFTER having defined the previous variables
# 3) Finally add the problem name in the 'get_helper' function

def get_helper(problem):
    if problem=='qap':
        return QAP_Experiment
    elif problem=='tsp':
        return TSP_Experiment
    elif problem=='tsprl':
        return TSP_RL_Experiment
    elif problem=='tspd':
        return TSP_Distance_Experiment
    elif problem=='tsppos':
        return TSP_Position_Experiment
    elif problem=='mcp':
        return MCP_Experiment
    elif problem=='mcptrue':
        return MCP_True_Experiment
    elif problem=='sbm':
        return SBM_Edge_Experiment
    else:
        raise NotImplementedError(f"Problem {problem} not implemented.")

'''
Object of the Experiment class is an evolved Experiment object, which knows which 
'''
class Experiment_Helper(Experiment): #Should not be called as such. Only its children should be called

    def __init__(self, problem, name, options=dict(), run=None, **kwargs) -> None:
        super(Experiment_Helper,self).__init__(name, options=options, run=run)
        self.problem = problem

        self.lr_threshold = options['train']['lr_stop']

        self.init_update_eval()
        self.init_loggers()

    def get_logger(self):
        logger = Experiment(self.name,self.options,self.run)
        logger.meters = copy.copy(self.meters)
        logger.logged = copy.copy(self.logged)
        logger.date_and_time = self.date_and_time
        return logger

    def init_loggers(self) -> None:
        """
        Initializes the internal loggers for the evaluation method
        Needs self.metric to be
        """
        metric = self.metric
        if metric=="acc":
            self.add_meters('train', metrics.make_meter_acc())
            self.add_meters('val', metrics.make_meter_acc())
            self.add_meters('test', metrics.make_meter_acc())
            self.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
        elif metric=="f1":
            self.add_meters('train', metrics.make_meter_f1())
            self.add_meters('val', metrics.make_meter_f1())
            self.add_meters('test', metrics.make_meter_f1())
            self.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
        elif metric=="loss":
            self.add_meters('train', metrics.make_meter_loss())
            self.add_meters('val', metrics.make_meter_loss())
            self.add_meters('test', metrics.make_meter_loss())
            self.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
        else:
            raise NotImplementedError(f"{metric} metric not implemented")

    def init_update_eval(self) -> None:
        """
        Initializes the update function for the evaluation method
        """
        metric = self.metric
        if metric=="acc":
            self.update_eval = self._update_meter_acc
        elif metric=="f1":
            self.update_eval = self._update_meter_f1
        elif metric=="loss":
            self.update_eval = self._update_meter_loss
        else:
            raise NotImplementedError(f"{metric} metric not implemented")
    
    def _update_meter_acc(self, name, values) -> None:
            '''
            name : 'train', 'val' or 'test'
            values : the values given by the eval function, in this case, should be of the form (true positives, total vertices explored)
            '''
            true_pos,n_total_vertices = values
            self.update_meter(name,'acc', true_pos, n=n_total_vertices)
    
    def _update_meter_f1(self, name, values) -> None:
            '''
            name : 'train', 'val' or 'test'
            values : the values given by the eval function, in this case, should be of the form (recall, precision, F1-score)
            '''
            precision, recall, f1_score = values
            self.update_meter(name,'precision', precision)
            self.update_meter(name,'recall', recall)
            self.update_meter(name,'f1', f1_score)
    
    def _update_meter_loss(self, name, values) -> None:
            '''
            name : 'train', 'val' or 'test'
            values : the values given by the eval function, in this case, should be of the form (loss)
            '''
            self.update_meter(name,'loss_ref', values)
    
    def get_eval_str(self, tag):
        string = ''
        if self.metric == 'acc':
            string =  'Acc {acc.avg:.3f} ({acc.val:.3f})'.format(acc = self.get_meter(tag,'acc'))
        elif self.metric == 'f1':
            f1 = self.get_meter(tag, 'f1')
            rec = self.get_meter(tag,'recall')
            prec = self.get_meter(tag,'precision')
            string = ('F1 {f1.avg:.3f} ({f1.val:.3f})\t'
                     'Recall {rec.avg:.3f} ({rec.val:.3f})\t'
                     'Prec {prec.avg:.3f} ({prec.val:.3f})').format(f1=f1,rec=rec,prec=prec)
        elif self.metric == 'loss':
            string =  'Loss Ref {loss_ref.avg:.3f} ({loss_ref.val:.3f})'.format(loss_ref = self.get_meter(tag,'loss_ref'))
        else:
            raise NotImplementedError(f"{self.metric} metric not implemented")
        
        return string

    def stop_condition(self, lr):
        return lr < self.lr_threshold

    def get_relevant_metric(self,tag):
        if self.metric=='acc':
            return self.get_meter(tag, 'acc')
        elif self.metric=='f1':
            return self.get_meter(tag, 'f1')
        elif self.metric=='loss':
            return self.get_meter(tag, 'loss_ref')
        else:
            raise NotImplementedError(f"{self.metric} metric not implemented")
    
    def get_relevant_metric_with_name(self,tag):
        if self.metric=='acc':
            return { 'acc' : self.get_meter(tag, 'acc').avg }
        elif self.metric=='f1':
            return { 'f1' : self.get_meter(tag, 'f1').avg }
        elif self.metric=='loss':
            return { 'loss_ref' : self.get_meter(tag, 'loss_ref').avg }
        else:
            raise NotImplementedError(f"{self.metric} metric not implemented")

    def to_json(self, log_dir, filename):
        utils.check_dir(log_dir)
        json_file = os.path.join(log_dir,filename)
        var_dict = copy.copy(vars(self))
        # Get rid of all non-json-able variables
        var_dict.pop('meters') 
        var_dict.pop('run')
        var_dict.pop('criterion')
        var_dict.pop('generator')
        var_dict.pop('eval_function')
        var_dict.pop('update_eval')
        #for key in ('viz', 'viz_dict'):
        #    if key in list(var_dict.keys()):
        #        var_dict.pop(key)    
        with open(json_file, 'w') as f:
            json.dump(var_dict, f, cls=utils.NpEncoder)



class QAP_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss_reduction='mean', loss=CrossEntropyLoss(reduction='sum')) -> None:
        self.generator = dg.QAP_Generator
        self.criterion = losses.triplet_loss(loss_reduction=loss_reduction, loss=loss)
        self.eval_function = metrics.accuracy_linear_assignment
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('qap', name, options=options, run=run)

class TSP_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize = Sigmoid) -> None:
        self.generator = dg.TSP_Generator
        self.criterion = losses.tsp_loss(loss=loss, normalize=normalize)
        self.eval_function = lambda raw_scores,target: metrics.compute_f1(raw_scores, target, k_best=2)
        
        self.metric = 'f1' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('tsp', name, options=options, run=run)

class TSP_RL_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, normalize = Softmax(dim=-1)) -> None:
        self.generator = dg.TSP_RL_Generator
        self.criterion = losses.tsp_rl_loss(normalize=normalize)
        self.eval_function = metrics.tsp_rl_loss
        
        self.metric = 'loss' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('tsp', name, options=options, run=run)

class TSP_Distance_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=MSELoss(reduction='none'), normalize = Sigmoid) -> None:
        self.generator = dg.TSP_Distance_Generator
        self.criterion = losses.tspd_loss(loss=loss, normalize=normalize)
        self.eval_function = metrics.tspd_dumb
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('tsp', name, options=options, run=run)

class TSP_Position_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize = Sigmoid) -> None:
        self.generator = dg.TSP_normpos_Generator
        self.criterion = losses.tsp_loss(loss=loss, normalize=normalize)
        self.eval_function = lambda raw_scores,target: metrics.compute_f1(raw_scores, target, k_best=2)
        
        self.metric = 'f1' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('tsp', name, options=options, run=run)


class MCP_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize=Sigmoid) -> None:
        self.generator = dg.MCP_Generator
        self.criterion = losses.mcp_loss(loss=loss, normalize=normalize)
        self.eval_function = metrics.accuracy_mcp
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super(MCP_Experiment,self).__init__('mcp', name, options=options, run=run)

class MCP_True_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize=Sigmoid) -> None:
        self.generator = dg.MCP_True_Generator
        self.criterion = losses.mcp_loss(loss=loss, normalize=normalize)
        self.eval_function = metrics.accuracy_mcp
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super(MCP_True_Experiment,self).__init__('mcp', name, options=options, run=run)

class SBM_Edge_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize=Sigmoid) -> None:
        
        self.generator = dg.SBM_Generator
        self.criterion = losses.sbm_edge_loss(loss=loss, normalize=normalize)
        self.eval_function = metrics.accuracy_sbm_two_categories_edge
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('sbm', name, options=options, run=run)

class SBM_Node_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=MSELoss(reduction='none'), normalize=Sigmoid) -> None:
        
        self.generator = dg.SBM_Generator
        self.criterion = losses.sbm_node_loss(loss=loss,normalize = normalize)
        self.eval_function = metrics.accuracy_sbm_two_categories
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('sbm', name, options=options, run=run)


