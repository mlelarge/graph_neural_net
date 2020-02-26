import os
import json
import torch
import torch.backends.cudnn as cudnn
import numpy as np

def setup_env(args):
    torch.manual_seed(args['--seed'])
    if not args['--cpu']:
        torch.cuda.manual_seed(args['--seed'])
        cudnn.benchmark = True

# create necessary folders and config files
def init_output_env(args):
    check_dir(os.path.join(args['--root_dir'],'runs'))
    check_dir(args['--log_dir'])
    for name in ['train', 'test', 'val']:
        check_dir(os.path.join(args['--path_dataset'], name))
    #check_dir(os.path.join(args.log_dir,'tensorboard'))
    #check_dir(args['--res_dir'])
    with open(os.path.join(args['--log_dir'], 'config.json'), 'w') as f:
        json.dump(args, f)


# check if folder exists, otherwise create it
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)   

# from https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
