import time
import json
import copy
import os
from collections import defaultdict
import toolbox.utils as utils
from loaders.data_generator import QAP_Generator,TSP_Generator,MCP_Generator
import toolbox.metrics as metrics
import toolbox.losses as losses
from torch.nn import BCELoss

'''
Object of the Experiment class keep track of scores and metrics across epochs.
This data is saved to json files after each epoch. 
'''
class Experiment(object):

    def __init__(self, name, options=dict(), run=None):
        """ Create an experiment
            run, if provided, must be a sacred Run object,
            to be used for logging
        """
        super(Experiment, self).__init__()

        self.name = name
        self.options = options
        self.date_and_time = time.strftime('%d-%m-%Y--%H-%M-%S')

        #self.info = defaultdict(dict)
        self.logged = defaultdict(dict)
        self.meters = defaultdict(dict)

        self.run = run

    def __getstate__(self):
        """ Overrides default pickle behaviour for this cls (run is not serializable) """
        # shallow copy
        d = self.__dict__.copy()
        d['run'] = None
        return d
        
    def add_meters(self, tag, meters_dict):
        assert tag not in (self.meters.keys())
        for name, meter in meters_dict.items():
            self.add_meter(tag, name, meter)

    def add_meter(self, tag, name, meter):
        assert name not in list(self.meters[tag].keys()), \
            "meter with tag {} and name {} already exists".format(tag, name)
        self.meters[tag][name] = meter

    def update_options(self, options_dict):
        self.options.update(options_dict)

    def update_meter(self, tag, name, val, n=1):
        meter = self.get_meter(tag, name).update(val, n=n)
    
    def update_value_meter(self, tag, name, val):
        meter = self.get_meter(tag, name).update(val)

    def log_meter(self, tag, name, n=1):
        meter = self.get_meter(tag, name)
        if name not in self.logged[tag]:
            self.logged[tag][name] = {}
        self.logged[tag][name][n] = meter.value()
        try:
            is_active = meter.is_active()
        except AttributeError:
            is_active = True
        if self.run and is_active:
            self.run.log_scalar("{}.{}".format(tag, name), meter.value())

    def log_meters(self, tag, n=1):
        for name, meter in self.get_meters(tag).items():
            self.log_meter(tag, name, n=n)

    def reset_meters(self, tag):
        meters = self.get_meters(tag)
        for name, meter in meters.items():
            meter.reset()

    def get_meters(self, tag):
        assert tag in list(self.meters.keys()), f"Tag {tag} not in {list(self.meters.keys())}"
        return self.meters[tag]

    def get_meter(self, tag, name):
        assert tag in list(self.meters.keys()), f"Tag {tag} not in {list(self.meters.keys())}"
        assert name in list(self.meters[tag].keys()), f"Name {name} not in {self.meters[tag].keys()}"
        return self.meters[tag][name]

    def to_json(self, log_dir, filename):
        utils.check_dir(log_dir)
        json_file = os.path.join(log_dir,filename)
        var_dict = copy.copy(vars(self))
        var_dict.pop('meters')
        var_dict.pop('run')
        #for key in ('viz', 'viz_dict'):
        #    if key in list(var_dict.keys()):
        #        var_dict.pop(key)    
        with open(json_file, 'w') as f:
            json.dump(var_dict, f, cls=utils.NpEncoder)

    def from_json(self, filename):
        with open(filename, 'r') as f:
            var_dict = json.load(f)
        xp = Experiment('')
        xp.date_and_time = var_dict['date_and_time']
        xp.logged        = var_dict['logged']
        
        #if 'info' in var_dict:
        #    xp.info          = var_dict['info']
        xp.options       = var_dict['options']
        xp.name          = var_dict['name']
        return xp

'''
Object of the Experiment class is an evolved Experiment object, which knows which 
'''
class Experiment_Helper(Experiment):

    def __init__(self, problem, name, options=dict(), run=None, **kwargs) -> None:
        super().__init__(name, options=options, run=run)
        self.problem = problem
        self.generator = self.get_generator(problem)
        self.eval_function = self.get_eval(problem)
        self.criterion = self.get_criterion(problem,**kwargs)
        
        self.init_metrics_handlers()

    def get_logger(self):
        logger = Experiment(self.name,self.options,self.run)
        logger.meters = copy.copy(self.meters)
        logger.logged = copy.copy(self.logged)
        logger.date_and_time = self.date_and_time
        return logger

    @classmethod
    def get_generator(cls,problem):
        """
        returns the problem associated generator
        """
        if problem=="qap":
            generator = QAP_Generator
        elif problem=="tsp":
            generator = TSP_Generator
        elif problem=="mcp":
            generator = MCP_Generator
        else:
            raise NotImplementedError(f"{problem} problem not implemented")
        return generator
    
    @classmethod
    def get_eval(cls,problem, device = 'cuda'):
        """
        returns the problem associated eval function
        """
        if problem=="qap":
            func = metrics.accuracy_max
        elif problem=="tsp":
            func = metrics.compute_f1
        elif problem=="mcp":
            func = metrics.accuracy_mcp
        else:
            raise NotImplementedError(f"{problem} problem not implemented")
        return func
    
    @classmethod
    def get_criterion(cls,problem, device = 'cuda', reduction='mean'):
        """
        returns the problem associated criterion
        """
        if problem=="qap":
            crit = losses.triplet_loss(device=device,loss_reduction=reduction)
        elif problem=="tsp":
            crit = losses.tsp_loss(loss = BCELoss(reduction=reduction))
        elif problem=="mcp":
            crit = BCELoss(reduction=reduction)
        else:
            raise NotImplementedError(f"{problem} problem not implemented")
        return crit

    def init_metrics_handlers(self) -> None:
        """
        Initializes all the functions depending on the metrics
        """
        problem = self.problem
        if problem=="qap":
            self.metric = 'acc'
        elif problem=="tsp":
            self.metric = 'f1'
        elif problem=="mcp":
            self.metric = 'acc'
        else:
            raise NotImplementedError(f"{problem} problem not implemented")
        
        self.init_update_eval()
        self.init_loggers()

    def init_loggers(self) -> None:
        """
        Initializes the internal loggers for the evaluation method
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
        else:
            raise NotImplementedError(f"{self.metric} metric not implemented")
        
        return string

    def get_relevant_metric(self,tag):
        if self.metric=='acc':
            return self.get_meter(tag, 'acc')
        elif self.metric=='f1':
            return self.get_meter(tag, 'f1')
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



