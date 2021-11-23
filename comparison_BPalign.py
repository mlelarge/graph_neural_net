import os
import shutil
from typing import Tuple
from torch._C import _logging_set_logger
import tqdm
import time
import torch
import toolbox.utils as utils
import neptune.new as neptune
from torch.utils.data import DataLoader
from commander import load_model
from toolbox.data_handler import Planner_Multi
from collections import namedtuple
from loaders.data_generator import QAP_Generator
from trainer import train_triplet, val_triplet
from toolbox.helper import QAP_Experiment
from models import get_model_gen


MODEL_NAME = "fgnn-comparison-l_{}-s_{}.tar"
DATA_PATH = "FGNN-BPA.csv"
LOG_DIR = "gnnbpa"
REMOVE_FILES_AFTER_EXP = True
TRAINING = True
TESTING  = True

cur_task = ""
if TRAINING:
    cur_task += "training "
if TESTING:
    cur_task += "testing"
if (not TRAINING) and (not TESTING):
    print("No current task, stopping.")
    exit()
print(f"Current task(s) : {cur_task}")

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

def round(t,dec=8):
    return torch.round(t*(10**dec))/(10**dec)

noise_bpa = torch.linspace(0.4,1,20)
noise_bpa = round(noise_bpa,8)
noise_gnn = 1 - noise_bpa
noise_gnn = round(noise_bpa,8)
l_lbda = [2,2.5,3]
n = 200

BATCH_SIZE = 32
MAX_EPOCHS = 100
START_LR = 1e-3

seed=23983892
torch.manual_seed(seed)

DEVICE = 'cuda' if (torch.cuda.is_available()) else 'cpu'
print('Using device:', DEVICE)

BASE_GEN_CONFIG = { 'num_examples_train': 10,
                'num_examples_val': 10,
                'num_examples_test': 10,
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

USE_NEPTUNE = True
global run
run=None

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

def _train_function(train_loader,model,optimizer,
                helper,device,epoch,print_freq=100):
    global run
    l_loss = []
    l_acc = []
    model.train()
    learning_rate = optimizer.param_groups[0]['lr']

    for i, (data, _) in enumerate(train_loader):

        data = data.to(device)
        output = model(data)#,input2)
        raw_scores = output.squeeze(-1)

        loss = helper.criterion(raw_scores,None)
        true_pos,n_tot = helper.eval_function(raw_scores,None)
        acc = true_pos/n_tot
        l_loss.append(loss.data.item())
        l_acc.append(acc)

        if USE_NEPTUNE:
            run['train/loss'].log(loss)
            run['train/acc'].log(acc)
            run['lr'].log(learning_rate)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR {lr:.2e}\t'
                  'Loss {loss:.4f} ({losavg:.4f})\t'
                  'acc : {acc} ({accavg})'.format(
                   epoch, i, len(train_loader),
                   lr=learning_rate,
                   loss=loss, losavg = sum(l_loss)/len(l_loss),
                   acc=acc, accavg=sum(l_acc)/len(l_acc)))
    optimizer.zero_grad()

def _val_function(val_loader,model,helper,device,epoch,print_freq=10,val_test='val'):
    l_loss = []
    l_acc = []
    global run
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):
            
            data = data.to(device)
            raw_scores = model(data)
            
            loss = helper.criterion(raw_scores,None)
            l_loss.append(loss.data.item())
            
            true_pos,n_tot = helper.eval_function(raw_scores,None)
            acc = true_pos/n_tot
            l_acc.append(acc)

            if USE_NEPTUNE:
                run[f'{val_test}/loss'].log(loss)
                run[f'{val_test}/acc'].log(acc)

            if i % print_freq == 0:
                if val_test == 'val':
                    val_test_string = "Validation"
                else:
                    val_test_string = "Test"
                print('{3} set, epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss:.4f} ({losavg:.4f})\t'
                        'Acc {acc} ({accavg})'.format(
                        epoch, i, len(val_loader), val_test_string,
                        loss=loss,losavg = sum(l_loss)/len(l_loss),
                        acc = acc, accavg = sum(l_acc)/len(l_acc)))

    return sum(l_acc)/len(l_acc),sum(l_loss)/len(l_loss)

def train_cycle(task):

    print("Starting training cycle.")
    best_score = 0

    train_gen = get_generator(*task,task='train')
    print("Loading train dataset... ", end = "")
    train_gen.load_dataset()
    print("Creating loader... ", end = "")
    train_loader = DataLoader(train_gen, BATCH_SIZE, shuffle=True)
    print("Done")
    val_gen = get_generator(*task,task='val')
    print("Loading val dataset... ", end = "")
    val_gen.load_dataset()
    print("Creating loader... ", end = "")
    val_loader = DataLoader(train_gen, BATCH_SIZE, shuffle=True)
    print("Done")

    model = get_model_gen(MODEL_CONFIG)
    model.to(DEVICE)
    model_name = MODEL_NAME.format(task.lbda,task.noise_gnn)

    optimizer = torch.optim.Adam(model.parameters(),lr=START_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **SCHEDULER_CONFIG)

    exp_helper = QAP_Experiment('comparisonGNNBPA', HELPER_OPTIONS)

    for epoch in range(MAX_EPOCHS):
        print('Current epoch: ', epoch)
        _train_function(train_loader,model,optimizer,exp_helper,DEVICE,epoch,print_freq=100)


        relevant_metric, loss = _val_function(val_loader,model,exp_helper,DEVICE,epoch)
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
    print("Starting test cycle...")
    model = get_model_gen(MODEL_CONFIG)
    model_name = MODEL_NAME.format(task.lbda,task.noise_gnn)
    model_full_path = os.path.join(LOG_DIR, 'best-' + model_name)
    if not os.path.exists(model_full_path):
        raise FileNotFoundError("Model {} not found".format(model_full_path))
    print("Loading model... ", end='')
    model = load_model(model, DEVICE, model_full_path)
    model.to(DEVICE)
    print("Model loaded.")

    test_gen = get_generator(*task,task='test')
    print("Preparing dataset... ", end="")
    test_gen.load_dataset()
    print("Creating dataloader... ", end = "")
    test_loader = DataLoader(test_gen, BATCH_SIZE, shuffle=True)
    print("Done")

    helper = QAP_Experiment('comparisonGNNBPA', HELPER_OPTIONS)
    
    print("Starting testing.")
    relevant_metric, loss = _val_function(test_loader, model, helper, DEVICE,
                                    epoch=0, val_test='test')
    if REMOVE_FILES_AFTER_EXP:
        print("Removing test files... ", end='')
        test_gen.remove_file()
        print("Files removed.")

    return relevant_metric


def one_exp(task):
    if USE_NEPTUNE:
        global run
        run = neptune.init(project='mautrib/bpagnn')
        run['task'] = task._asdict()

    model_name = MODEL_NAME.format(task.lbda,task.noise_gnn)
    model_full_path = os.path.join(LOG_DIR, 'best-' + model_name)
    if not os.path.exists(model_full_path):
        if TRAINING:
            train_cycle(task)
    if TESTING and os.path.exists(model_full_path):
        relevant_metric = test_cycle(task)
    else:
        print(f"Model for task {task} not found, skipping.")
    if USE_NEPTUNE:
        run.stop()
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
    else:
        print(f"{n_tasks} tasks to be done.")
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



