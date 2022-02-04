import time
import json
import copy
import os
from collections import defaultdict
import toolbox.utils as utils
from loaders.data_generator import QAP_Generator #, SBM_Generator,TSP_Generator,MCP_Generator
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