import os
import shutil
import torch
from torch.utils.data import DataLoader
from commander import load_model
from toolbox.data_handler import Planner_Multi
from collections import namedtuple
import tqdm
from loaders.data_generator import QAP_Generator
from trainer import train_triplet, val_triplet
from toolbox.helper import QAP_Experiment
from models import get_model_gen
import toolbox.utils as utils


MODEL_NAME = "fgnn-comparison-l_{}-s_{}.tar"
DATA_PATH = "FGNN-BPA.csv"
LOG_DIR = "gnnbpa"
REMOVE_FILES_AFTER_EXP = True
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


noise_bpa = torch.linspace(0.4,1,20)
noise_gnn = 1 - noise_bpa
l_lbda = [2,2.5,3]
n = 200

BATCH_SIZE = 32
MAX_EPOCHS = 100
START_LR = 1e-3

seed=23983892
torch.manual_seed(seed)

DEVICE = 'cuda' if (torch.cuda.is_available()) else 'cpu'
print('Using device:', DEVICE)

BASE_GEN_CONFIG = { 'num_examples_train': 10000,
                'num_examples_val': 1000,
                'num_examples_test': 1000,
                'generative_model': 'ErdosRenyi', # so far ErdosRenyi, Regular or BarabasiAlbert
                'noise_model': 'ErdosRenyi',
                'vertex_proba': 1., # Parameter of the binomial distribution of vertices
                'path_dataset': 'dataset_qap' # Path where datasets are stored
} #'edge_density' and 'noise' to be defined later
MODEL_CONFIG = {
    'arch': 'fgnn', #fgnn or gcn
    'arch_load': 'siamese', #siamese or simple
    'embedding': 'node', #node or edge
    'num_blocks': 2,
    'original_features_num': 2,
    'in_features': 64,
    'out_features': 64,
    'depth_of_mlp': 3
}
HELPER_OPTIONS = {
    'train': {'lr_stop': 1e-7}
}
SCHEDULER_CONFIG = {
    'factor': 0.5,
    'patience': 2,
    'verbose': True
}

Task = namedtuple('Task',['lbda','noise_bpa','noise_gnn','n'])


def get_generator(lbda,nbpa,ngnn,n,task='train'):
    exp_dict = BASE_GEN_CONFIG.copy()
    exp_dict['n_vertices'] = n
    exp_dict['noise'] = ngnn
    exp_dict['edge_density'] = lbda/n
    gen = QAP_Generator(task,exp_dict)
    return gen

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    utils.check_dir(log_dir)
    torch.save(state, os.path.join(log_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(log_dir,filename), os.path.join(log_dir, 'best-' + filename))
        print(f"Best Model yet : saving at {log_dir+'best-' + filename}")

    fn = os.path.join(log_dir, 'checkpoint_epoch{}.pth.tar')
    torch.save(state, fn.format(state['epoch']))

    if (state['epoch'] - 1 ) % 5 != 0:
      #remove intermediate saved models, e.g. non-modulo 5 ones
      if os.path.exists(fn.format(state['epoch'] - 1 )):
          os.remove(fn.format(state['epoch'] - 1 ))

    state['exp_logger'].to_json(log_dir=log_dir,filename='logger.json')


def train_cycle(task):
    best_score = 0

    train_gen = get_generator(*task,task='train')
    train_gen.load_dataset()
    train_loader = DataLoader(train_gen, BATCH_SIZE, shuffle=True)
    val_gen = get_generator(*task,task='val')
    val_gen.load_dataset()
    val_loader = DataLoader(train_gen, BATCH_SIZE, shuffle=True)

    model = get_model_gen(MODEL_CONFIG)
    model_name = MODEL_NAME.format(task.lbda,task.noise_gnn)

    optimizer = torch.optim.Adam(model.parameters(),lr=START_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **SCHEDULER_CONFIG)

    exp_helper = QAP_Experiment('comparisonGNNBPA', HELPER_OPTIONS)

    for epoch in range(MAX_EPOCHS):
        print('Current epoch: ', epoch)
        train_triplet(train_loader,model,optimizer,exp_helper,DEVICE,epoch,eval_score=True,print_freq=100)
        


        relevant_metric, loss = val_triplet(val_loader,model,exp_helper,DEVICE,epoch,eval_score=True)
        scheduler.step(loss)
        # remember best acc and save checkpoint
        #TODO : if relevant metric is like a loss and needs to decrease, this doesn't work
        is_best = (relevant_metric > best_score)
        best_score = max(relevant_metric, best_score)
        if is_best:
            best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'exp_logger': exp_helper.get_logger(),
            }, is_best, LOG_DIR, model_name)

        cur_lr = utils.get_lr(optimizer)
        if exp_helper.stop_condition(cur_lr):
            print(f"Learning rate ({cur_lr}) under stopping threshold, ending training.")
            break
    if REMOVE_FILES_AFTER_EXP:
        print("Removing train and val files... ", end='')
        train_gen.remove_file()
        val_gen.remove_file()
        print("Files removed.")


def test_cycle(task):
    model = get_model_gen(MODEL_CONFIG)
    model_name = MODEL_NAME.format(task.lbda,task.noise_gnn)
    model_full_path = os.path.join(LOG_DIR, 'best-' + model_name)
    if not os.path.exists(model_full_path):
        raise FileNotFoundError("Model {} not found".format(model_full_path))
    model = load_model(model, DEVICE, model_full_path)

    test_gen = get_generator(*task,task='test')
    test_gen.load_dataset()
    test_loader = DataLoader(test_gen, BATCH_SIZE, shuffle=True)

    helper = QAP_Experiment('comparisonGNNBPA', HELPER_OPTIONS)
    
    relevant_metric, loss = val_triplet(test_loader, model, helper, DEVICE,
                                    epoch=0, eval_score=True,
                                    val_test='test')
    if REMOVE_FILES_AFTER_EXP:
        print("Removing test files... ", end='')
        test_gen.remove_file()
        print("Files removed.")

    return relevant_metric


def one_exp(task):
    model_name = MODEL_NAME.format(task.lbda,task.noise_gnn)
    model_full_path = os.path.join(LOG_DIR, 'best-' + model_name)
    if not os.path.exists(model_full_path):
        train_cycle(task)
    relevant_metric = test_cycle(task)
    return relevant_metric
    


def main():
    FULL_PATH = os.path.join(LOG_DIR,DATA_PATH)
    planner = Planner_Multi(FULL_PATH)
    planner.new_columns(["lbda","noise_bpa","noise_gnn","n","overlap"])

    for lbda in l_lbda:
        for nbpa,ngnn in zip(noise_bpa, noise_gnn):
            planner.add_task(Task(lbda,float(nbpa),float(ngnn),n))

    n_tasks = planner.n_tasks

    if n_tasks==0:
        print("No tasks to be done, ending.")
    
    progress_bar = tqdm.trange(n_tasks)
    i = 0
    for _ in progress_bar:
        i+=1
        task = planner.next_task()
        progress_bar.set_description(f"task {i}/{n_tasks}: {task}")
        relevant_metric = one_exp(task)
        progress_bar.set_description(f"Finished task {i}/{n_tasks}: {task}")
        task_as_dict = task._asdict()
        task_as_dict['overlap'] = relevant_metric
        planner.add_entry(task_as_dict,save = True)

if __name__=='__main__':
    main()



