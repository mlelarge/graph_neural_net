import os
import pathlib
import shutil
import json
from sacred import Experiment

import torch
import torch.backends.cudnn as cudnn
from toolbox import logger, metrics
from models import get_model
from loaders.siamese_loaders import siamese_loader
from loaders.data_generator import Generator
from toolbox.optimizer import get_optimizer
from toolbox.losses import get_criterion
from toolbox import utils
import trainer as trainer

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

### BEGIN Sacred setup
ex = Experiment()
ex.add_config('default_qap.yaml')

@ex.config_hook
def set_experiment_name(config, command_name, logger):
    ex.path = config['name']
    return config

@ex.config
def update_paths(root_dir, name, train_data, test_data):
    log_dir = '{}/runs/{}/QAP_{}_{}_{}_{}_{}_{}/'.format(
              root_dir, name, train_data['generative_model'],
              train_data['noise_model'], train_data['n_vertices'],
              train_data['vertex_proba'], train_data['noise'],
              train_data['edge_density'])
    path_dataset = train_data['path_dataset']
    # The two keys below are specific to testing
    # These default values are overriden by command line
    model_path = os.path.join(log_dir, 'model_best.pth.tar')
    output_filename = 'test.json'

@ex.config_hook
def init_observers(config, command_name, logger):
    if command_name == 'train':
        neptune = config['observers']['neptune']
        if neptune['enable']:
            from neptunecontrib.monitoring.sacred import NeptuneObserver
            ex.observers.append(NeptuneObserver(project_name=neptune['project']))
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
def init_logger(name, _config, _run):
    # set loggers
    exp_logger = logger.Experiment(name, _config, run=_run)
    exp_logger.add_meters('train', metrics.make_meter_matching())
    exp_logger.add_meters('val', metrics.make_meter_matching())
    #exp_logger.add_meters('test', metrics.make_meter_matching())
    exp_logger.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
    return exp_logger
 
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

    fn = os.path.join(log_dir, 'checkpoint_epoch{}.pth.tar')
    torch.save(state, fn.format(state['epoch']))

    if (state['epoch'] - 1 ) % 5 != 0:
      #remove intermediate saved models, e.g. non-modulo 5 ones
      if os.path.exists(fn.format(state['epoch'] - 1 )):
          os.remove(fn.format(state['epoch'] - 1 ))

    state['exp_logger'].to_json(log_dir=log_dir,filename='logger.json')


@ex.command
def train(cpu, train_data, train, arch, log_dir):
    """ Main func.
    """
    global best_score, best_epoch
    best_score, best_epoch = -1, -1
    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    setup_env()
    print("Models saved in ", log_dir)

    init_output_env()
    exp_logger = init_logger()
    
    gene_train = Generator('train', train_data)
    gene_train.load_dataset()
    train_loader = siamese_loader(gene_train, train['batch_size'],
                                  gene_train.constant_n_vertices)
    gene_val = Generator('val', train_data)
    gene_val.load_dataset()
    val_loader = siamese_loader(gene_val, train['batch_size'],
                                gene_val.constant_n_vertices)
    
    model = get_model(arch)
    optimizer, scheduler = get_optimizer(train,model)
    criterion = get_criterion(device, train['loss_reduction'])

    model.to(device)

    is_best = True
    for epoch in range(train['epoch']):
        print('Current epoch: ', epoch)
        trainer.train_triplet(train_loader,model,criterion,optimizer,exp_logger,device,epoch,eval_score=metrics.accuracy_max,print_freq=train['print_freq'])
        


        acc, loss = trainer.val_triplet(val_loader,model,criterion,exp_logger,device,epoch,eval_score=metrics.accuracy_linear_assignment)
        scheduler.step(loss)
        # remember best acc and save checkpoint
        is_best = (acc > best_score)
        best_score = max(acc, best_score)
        if True == is_best:
            best_epoch = epoch
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'exp_logger': exp_logger,
            }, is_best)

### Testing
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

def save_to_json(key, acc, loss, filename):
    if os.path.exists(filename):
        with open(filename, "r") as jsonFile:
            data = json.load(jsonFile)
    else:
        data = {}
    data[key] = {'loss':loss, 'acc': acc}
    with open(filename, 'w') as jsonFile:
        json.dump(data, jsonFile)

@ex.capture
def create_key(log_dir, test_data):
    template = 'model_{}data_QAP_{}_{}_{}_{}_{}_{}_{}'
    key=template.format(log_dir, test_data['generative_model'], test_data['noise_model'],
                        test_data['num_examples_test'], test_data['n_vertices'],
                        test_data['vertex_proba'], test_data['noise'],
                        test_data['edge_density'])
    return key

@ex.command
def eval(name, cpu, test_data, train, arch, log_dir, model_path, output_filename):
    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    model = get_model(arch)
    model.to(device)
    model = load_model(model, device)

    criterion = get_criterion(device, train['loss_reduction'])
    exp_logger = logger.Experiment(name)
    exp_logger.add_meters('test', metrics.make_meter_matching())

    gene_test = Generator('test', test_data)
    gene_test.load_dataset()
    test_loader = siamese_loader(gene_test, train['batch_size'],
                                 gene_test.constant_n_vertices)
    acc, loss = trainer.val_triplet(test_loader, model, criterion, exp_logger, device,
                                    epoch=0, eval_score=metrics.accuracy_linear_assignment,
                                    val_test='test')
    key = create_key()
    filename_test = os.path.join(log_dir,  output_filename)
    print('Saving result at: ',filename_test)
    save_to_json(key, acc, loss, filename_test)

@ex.automain
def main():
    pass
