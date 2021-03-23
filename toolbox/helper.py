import json
import copy
import os

from torch.nn import BCELoss, CrossEntropyLoss, Sigmoid
from torch.nn.modules.activation import Softmax
from toolbox.logger import Experiment
from loaders.data_generator import QAP_Generator,TSP_Generator,TSP_RL_Generator, MCP_Generator,SBM_Generator
import toolbox.metrics as metrics
import toolbox.losses as losses
import toolbox.utils as utils


def get_helper(problem):
    if problem=='qap':
        return QAP_Experiment
    elif problem=='tsp':
        return TSP_Experiment
    elif problem=='tsprl':
        return TSP_RL_Experiment
    elif problem=='mcp':
        return MCP_Experiment
    elif problem=='sbm':
        return SBM_Experiment
    else:
        raise NotImplementedError(f"Problem {problem} not implemented.")

'''
Object of the Experiment class is an evolved Experiment object, which knows which 
'''
class Experiment_Helper(Experiment): #Should not be called as such. Only its children should be called

    def __init__(self, problem, name, options=dict(), run=None, **kwargs) -> None:
        super(Experiment_Helper,self).__init__(name, options=options, run=run)
        self.problem = problem

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

    def get_relevant_metric(self,tag):
        if self.metric=='acc':
            return self.get_meter(tag, 'acc')
        elif self.metric=='f1':
            return self.get_meter(tag, 'f1')
        elif self.metric=='loss':
            return self.get_meter(tag, 'loss_ref')
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
        self.generator = QAP_Generator
        self.criterion = losses.triplet_loss(loss_reduction=loss_reduction, loss=loss)
        self.eval_function = metrics.accuracy_max
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('qap', name, options=options, run=run)

class TSP_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize = Sigmoid()) -> None:
        self.generator = TSP_Generator
        self.criterion = losses.tsp_loss(loss=loss, normalize=normalize)
        self.eval_function = metrics.compute_f1_3
        
        self.metric = 'f1' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('tsp', name, options=options, run=run)

class TSP_RL_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, normalize = Softmax(dim=-1)) -> None:
        self.generator = TSP_RL_Generator
        self.criterion = losses.tsp_rl_loss(normalize=normalize)
        self.eval_function = metrics.tsp_rl_loss
        
        self.metric = 'loss' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('tsp', name, options=options, run=run)


class MCP_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize=Sigmoid()) -> None:
        self.generator = MCP_Generator
        self.criterion = losses.mcp_loss(loss=loss, normalize=normalize)
        self.eval_function = metrics.accuracy_mcp
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super(MCP_Experiment,self).__init__('mcp', name, options=options, run=run)

class SBM_Experiment(Experiment_Helper):
    def __init__(self, name, options=dict(), run=None, loss=BCELoss(reduction='none'), normalize=Sigmoid()) -> None:
        self.generator = SBM_Generator
        self.criterion = losses.sbm_loss(loss=loss, normalize=normalize)
        self.eval_function = metrics.accuracy_sbm
        
        self.metric = 'acc' #Will be used in super() to compute the relevant metric meter, printer function and update_eval for the logger function
        super().__init__('sbm', name, options=options, run=run)


