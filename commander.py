import os
#import shutil
import json
from sacred import Experiment
from sacred.config import config_scope
#import yaml

import torch
import torch.backends.cudnn as cudnn
from models import get_model_gen
from loaders.siamese_loaders import get_loader
from toolbox.optimizer import get_optimizer
import toolbox.utils as utils
import trainer as trainer
from toolbox.helper import get_helper
from datetime import datetime
from pathlib import Path

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ROOT_DIR = Path.home()
QAP_DIR = os.path.join(ROOT_DIR,'experiments-gnn/qap/')
DATA_QAP_DIR = os.path.join(QAP_DIR,'data/')

### BEGIN Sacred setup
ex = Experiment()

ex.add_config('default_config.yaml')

@ex.config_hook
def check_paths_qap(config, command_name, logger):
    """
    add to the configuration:
        'path_log' = root/experiments/qap/name_expe/
            (arch_gnn)_(numb_locks)_(generative_model)_(n_vertices)_(edge_density)/date_time
        'date_time' 
        ['data']['path_dataset'] = root/experiments/qap/data
    save the new configuration at path_log/config.json
    """ 
    assert config['problem'] == 'qap', f"paths only defined for QAP"
    assert config['use_dgl'] or config['arch']['arch_gnn'] not in ['gcn', 'gatedgcn'], f"use dgl for this architecture"
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%y-%H-%M")
    dic = {'date_time' : date_time}
    #expe_runs_qap = os.path.join(QAP_DIR,config['name'],'runs/')
    l_name = [config['arch']['arch_gnn'], config['arch']['num_blocks'],
        config['data']['train']['generative_model'], config['data']['train']['n_vertices'],
        config['data']['train']['edge_density']]
    name = "_".join([str(e) for e in l_name])
    name = os.path.join(name, str(date_time))
    path_log = os.path.join(QAP_DIR,config['name'], name)
    utils.check_dir(path_log)
    dic['path_log'] = path_log

    utils.check_dir(DATA_QAP_DIR)
    dic['data'] = {'path_dataset' : DATA_QAP_DIR}
    
    #print(path_log)
    config.update(dic)
    with open(os.path.join(path_log, 'config.json'), 'w') as f:
        json.dump(config, f)
    return config

@ex.config_hook
def set_experiment_name(config, command_name, logger):
    ex.path = config['name']
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
def init_helper(problem, name, _config, _run):
    exp_helper_object = get_helper(problem)
    exp_helper = exp_helper_object(name, _config, run=_run)
    return exp_helper
 
@ex.command
def train(cpu, train, problem, arch, test_enabled, path_log, use_dgl, data):

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
    utils.setup_env(cpu)
    
    print("Models saved in ", path_log)
    exp_helper = init_helper(problem)
    
    # rawscores need to be adapted for dgl
    if use_dgl:
        symmetric_problem=True
        print(f"Arch : {arch['arch_gnn']}")
        from loaders.siamese_loaders import get_uncollate_function
        uncollate_function = get_uncollate_function(data['train']['n_vertices'],problem)
        
    
    generator = exp_helper.generator
    gene_train = generator('train', data['train'], data['path_dataset'])
    gene_train.load_dataset(use_dgl)
    gene_val = generator('val', data['train'], data['path_dataset'])
    gene_val.load_dataset(use_dgl)
    train_loader = get_loader(use_dgl,gene_train, train['batch_size'],
                                  gene_train.constant_n_vertices,problem=problem,
                                  sparsify=data['train']['sparsify'])
    val_loader = get_loader(use_dgl,gene_val, train['batch_size'],
                                gene_val.constant_n_vertices,problem=problem,
                                sparsify=data['train']['sparsify'])
    
    model = get_model_gen(arch)
    optimizer, scheduler = get_optimizer(train,model)
    print("Model #parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if not train['anew']:
        try:
            utils.load_model(model,device,train['start_model'])
            print("Model found, using it.")
        except RuntimeError:
            print("Model not existing. Starting from scratch.")

    model.to(device)

    is_best = True
    try:
        for epoch in range(train['epoch']):
            print('Current epoch: ', epoch)
            if not use_dgl:
                trainer.train_triplet(train_loader,model,optimizer,exp_helper,device,epoch,eval_score=True,print_freq=train['print_freq'])
            else:
                trainer.train_triplet_dgl(train_loader,model,optimizer,exp_helper,device,epoch,uncollate_function,
                                            sym_problem=symmetric_problem,eval_score=True,print_freq=train['print_freq'])
            

            if not use_dgl:
                relevant_metric, loss = trainer.val_triplet(val_loader,model,exp_helper,device,epoch,eval_score=True)
            else:
                relevant_metric, loss = trainer.val_triplet_dgl(val_loader,model,exp_helper,device,epoch,uncollate_function,eval_score=True)
            scheduler.step(loss)
            # remember best acc and save checkpoint
            is_best = (relevant_metric > best_score)
            best_score = max(relevant_metric, best_score)
            if True == is_best:
                best_epoch = epoch
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'best_epoch': best_epoch,
                'exp_logger': exp_helper.get_logger(),
                }, is_best,path_log)

            cur_lr = utils.get_lr(optimizer)
            if exp_helper.stop_condition(cur_lr):
                print(f"Learning rate ({cur_lr}) under stopping threshold, ending training.")
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    if test_enabled:
        eval(use_model=model)


@ex.command
def eval(cpu, train, arch, data, use_dgl, problem, use_model=None):
    print("Heading to evaluation.")

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    if use_model is None:
        model = get_model_gen(arch)
        model.to(device)
        model = utils.load_model(model, device, train['start_model'])
    else:
        model = use_model
    
    helper = init_helper(problem)

    if use_dgl:
        print(f"Arch : {arch['arch_gnn']}")
        from loaders.siamese_loaders import get_uncollate_function
        uncollate_function = get_uncollate_function(data['test']['n_vertices'],problem)
        cur_crit = helper.criterion
        cur_eval = helper.eval_function
        helper.criterion = lambda output, target : cur_crit(uncollate_function(output), target)
        helper.eval_function = lambda output, target : cur_eval(uncollate_function(output), target)


    gene_test = helper.generator('test', data['test'], data['path_dataset'])
    gene_test.load_dataset(use_dgl)
    test_loader = get_loader(use_dgl,gene_test, train['batch_size'],
                                 gene_test.constant_n_vertices,problem=problem)
    
    relevant_metric, loss = trainer.val_triplet(test_loader, model, helper, device,
                                    epoch=0, eval_score=True,
                                    val_test='test')
    
    #key = create_key()
    #filename_test = os.path.join(log_dir,  output_filename)
    #print('Saving result at: ',filename_test)
    #metric_to_save = helper.get_relevant_metric_with_name('test')
    #utils.save_to_json(key, loss, metric_to_save, filename_test)

@ex.automain
def main():
    print("Main does nothing ! Use 'train' or 'eval' argument.")

