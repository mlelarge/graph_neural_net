import os
import shutil
import json
from sacred import Experiment
from sacred.config import config_scope
import yaml

import torch
import torch.backends.cudnn as cudnn
from models import get_model,get_model_gen
from loaders.siamese_loaders import get_loader
from toolbox.optimizer import get_optimizer
import toolbox.utils as utils
import trainer as trainer
from toolbox.helper import get_helper

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

### BEGIN Sacred setup
ex = Experiment()

#ex.add_config('default_config.yaml')

@ex.config
def create_config():
    with open("default_config.yaml") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    del f
    pbm = config['problem']
    pbm = utils.reduce_name(pbm)
    pbm_key='_'+pbm
    config = utils.clean_config(config,pbm_key)
    ex.add_config(config)


@ex.config_hook
def set_experiment_name(config, command_name, logger):
    ex.path = config['name']
    return config

@ex.config_hook
def update_paths(config, command_name, logger):
    pbm = config['problem']
    pbm = utils.reduce_name(pbm)

    pbm_key = "_"+pbm
    l_params = [config['data']['train']['n_vertices']]
    l_params += [value for _,value in (config['data']['train'][pbm_key]).items()]
    config_str = "_".join([str(item) for item in l_params])
    final_model_name = config['train']['model_name']
    log_dir = '{}/runs/{}/{}-{}/'.format(config['root_dir'], config['name'],
                                         config['problem'].upper(),
                                         config_str)
    config.update(  log_dir=log_dir, #End log_dir
                    path_dataset=config['data']['train'][pbm_key]['path_dataset'],
                    model_path = os.path.join(log_dir, final_model_name) if config['train']['anew']  else os.path.join(config['train']['template_model_path']),
                    output_filename = 'test.json'
    )
    return config

@ex.config_hook
def create_data_dict(config, command_name, logger):
    """
    Creates the parameter dictionaries for the data generation
    """
    problem = config['problem']
    problem = utils.reduce_name(problem)
    problem_key = "_" + problem
    train_config = config['data']['train']
    test_config = config['data']['test']
    custom_test = config['test_enabled'] and config['data']['test']['custom']
    assert problem_key in list(train_config.keys())
    train_data_dict = dict()
    test_data_dict = dict()
    for key,value in train_config.items():
        if key[0]!='_': #Don't add the problem data yet
            train_data_dict[key] = value
            if not custom_test:
                test_data_dict[key] = value
            else:
                try:
                    test_data_dict[key] = test_config[key]
                except KeyError: #In case the key doesn't exist in the custom test data, add it from the train anyways
                    test_data_dict[key] = value
        elif key==problem_key:#Add problem data
            for pb_key,pb_value in train_config[key].items():
                train_data_dict[pb_key] = pb_value
                if not custom_test:
                    test_data_dict[pb_key] = pb_value
                else:
                    try:
                        test_data_dict[pb_key] = test_config[key][pb_key]
                    except KeyError:
                        test_data_dict[pb_key] = pb_value

    
    test_data_dict['num_examples_test'] = test_config['num_examples_test'] #Special case of the numer of examples

    config.update(
        test_data_dict=test_data_dict,
        train_data_dict=train_data_dict
    )
    return config

@ex.config_hook
def init_observers(config, command_name, logger):
    neptune = config['observers']['neptune']
    problem = config['problem']
    if neptune['enable']:
        from neptunecontrib.monitoring.sacred import NeptuneObserver
        ex.observers.append(NeptuneObserver(project_name=neptune['project']+problem.upper()))

    return config

@ex.post_run_hook
def clean_observer(observers):
    """ Observers that are added in a config_hook need to be cleaned """
    try:
        neptune = observers['neptune']
        if neptune['enable']:
            from neptunecontrib.monitoring.sacred import NeptuneObserver
            ex.observers = [obs for obs in ex.observers
                            if not isinstance(obs, NeptuneObserver)]
    except KeyError:
        pass

### END Sacred setup

### Training
@ex.capture
def init_helper(problem, name, _config, _run, type='train'):
    exp_helper_object = get_helper(problem)
    exp_helper = exp_helper_object(name, _config, run=_run)
    return exp_helper
 
@ex.capture
def setup_env(cpu):
    # Randomness is already controlled by Sacred
    # See https://sacred.readthedocs.io/en/stable/randomness.html
    if not cpu:
        cudnn.benchmark = True

# create necessary folders and config files
@ex.capture
def init_output_env(_config, root_dir, log_dir, path_dataset):
    utils.check_dir(os.path.join(root_dir,'runs'))
    utils.check_dir(log_dir)
    utils.check_dir(path_dataset)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(_config, f)

@ex.capture
def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    utils.check_dir(log_dir)
    filename = os.path.join(log_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(log_dir, 'model_best.pth.tar'))
        print(f"Best Model yet : saving at {log_dir+'model_best.pth.tar'}")

    fn = os.path.join(log_dir, 'checkpoint_epoch{}.pth.tar')
    torch.save(state, fn.format(state['epoch']))

    if (state['epoch'] - 1 ) % 5 != 0:
      #remove intermediate saved models, e.g. non-modulo 5 ones
      if os.path.exists(fn.format(state['epoch'] - 1 )):
          os.remove(fn.format(state['epoch'] - 1 ))

    state['exp_logger'].to_json(log_dir=log_dir,filename='logger.json')



@ex.capture
def load_model(model, device, model_path):
    """ Load model. Note that the model_path argument is captured """
    if os.path.exists(model_path):
        print("Reading model from ", model_path)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        raise RuntimeError('Model does not exist!')

@ex.command
def train(cpu, train, problem, train_data_dict, arch, test_enabled,log_dir):

    """ Main func.
    """
    print("Heading to Training.")
    global best_score, best_epoch
    best_score, best_epoch = -1, -1
    print("Current problem : ", problem)

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    setup_env()
    print("Models saved in ", log_dir)

    init_output_env()
    

    if problem == 'mcp' and not train_data_dict['planted']:
        problem = 'mcptrue'
    exp_helper = init_helper(problem)

    if arch!='fgnn':
        from loaders.siamese_loaders import get_uncollate_function
        uncollate_function = get_uncollate_function(train_data_dict["n_vertices"])
        exp_helper.criterion = lambda output, target : exp_helper.criterion(uncollate_function(output), target)
    
    generator = exp_helper.generator
    
    gene_train = generator('train', train_data_dict)
    gene_train.load_dataset()

    gene_val = generator('val', train_data_dict)
    gene_val.load_dataset()
    
    train_loader = get_loader(arch['arch'],gene_train, train['batch_size'],
                                  gene_train.constant_n_vertices)
    val_loader = get_loader(arch['arch'],gene_val, train['batch_size'],
                                gene_val.constant_n_vertices)
    
    model = get_model_gen(arch)
    optimizer, scheduler = get_optimizer(train,model)
    print("Model #parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if not train['anew']:
        try:
            load_model(model,device)
            print("Model found, using it.")
        except RuntimeError:
            print("Model not existing. Starting from scratch.")

    model.to(device)

    is_best = True
    for epoch in range(train['epoch']):
        print('Current epoch: ', epoch)
        trainer.train_triplet(train_loader,model,optimizer,exp_helper,device,epoch,eval_score=True,print_freq=train['print_freq'])
        


        relevant_metric, loss = trainer.val_triplet(val_loader,model,exp_helper,device,epoch,eval_score=True)
        scheduler.step(loss)
        # remember best acc and save checkpoint
        #TODO : if relevant metric is like a loss and needs to decrease, this doesn't work
        is_best = (relevant_metric > best_score)
        best_score = max(relevant_metric, best_score)
        if True == is_best:
            best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'exp_logger': exp_helper.get_logger(),
            }, is_best)

        cur_lr = utils.get_lr(optimizer)
        if exp_helper.stop_condition(cur_lr):
            print(f"Learning rate ({cur_lr}) under stopping threshold, ending training.")
            break
    if test_enabled:
        eval(use_model=model)


@ex.capture
def create_key(problem, test_data_dict, _config):
    problem = utils.reduce_name(problem)
    relevant_keys = _config['data']['test'][f'_{problem}'].keys()
    relevant_params = '_'.join( [ str(test_data_dict[key]) for key in relevant_keys ] )

    problem = problem.upper()
    template = 'model_data_{}-{}_{}_{}'


    key=template.format(problem,
                        test_data_dict["num_examples_test"],
                        test_data_dict['n_vertices'],
                        relevant_params)
    return key

def save_to_json(jsonkey, loss, relevant_metric_dict, filename):

    if os.path.exists(filename):
        with open(filename, "r") as jsonFile:
            data = json.load(jsonFile)
    else:
        data = {}

    data[jsonkey] = {'loss':loss}

    for dkey, value in relevant_metric_dict.items():
        data[jsonkey][dkey] = value
    with open(filename, 'w') as jsonFile:
        json.dump(data, jsonFile)

@ex.command
def eval(cpu, test_data_dict, train, arch, log_dir, output_filename, problem, use_model=None):
    print("Heading to evaluation.")

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    if use_model is None:
        model = get_model(arch)
        model.to(device)
        model = load_model(model, device)
    else:
        model = use_model


    if problem == 'tsprl': #In case we're on RL TSP, we want to compare with a normal TSP at the end
        problem = 'tsp'
    elif problem == 'mcp' and not test_data_dict['planted']:
        problem = 'mcptrue'
    
    helper = init_helper(problem)

    gene_test = helper.generator('test', test_data_dict)
    gene_test.load_dataset()

    test_loader = get_loader(arch['arch'],gene_test, train['batch_size'],
                                 gene_test.constant_n_vertices)
    
    relevant_metric, loss = trainer.val_triplet(test_loader, model, helper, device,
                                    epoch=0, eval_score=True,
                                    val_test='test')
    
    key = create_key()
    filename_test = os.path.join(log_dir,  output_filename)
    print('Saving result at: ',filename_test)
    metric_to_save = helper.get_relevant_metric_with_name('test')
    save_to_json(key, loss, metric_to_save, filename_test)

@ex.automain
def main():
    print("Main does nothing ! Use 'train' or 'eval' argument.")

