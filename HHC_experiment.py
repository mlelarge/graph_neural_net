from loaders.data_generator import HHC_Generator
import toolbox.utils as utils
import tqdm
import torch
import numpy as np
import os

from commander import get_model,init_helper
from toolbox.optimizer import get_optimizer
from loaders.siamese_loaders import siamese_loader
from trainer import train_triplet,val_triplet
from toolbox.metrics import accuracy_hhc,perf_hhc

import sklearn.metrics as skmetrics

def add_line(filename,line) -> None:
    with open(filename,'a') as f:
        f.write(line + '\n')

def check_model_exists(model_path,n_vertices,mu)->bool:
    model_filename = os.path.join(model_path,f'model-n_{n_vertices}-mu_{mu}.tar')
    return os.path.isfile(model_filename)

def save_model(model_path,model,n_vertices,mu)->None:
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,f'model-n_{n_vertices}-mu_{mu}.tar')
    torch.save(model.state_dict(), model_filename)

def load_model_dict(model_path,n_vertices,mu):
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,f'model-n_{n_vertices}-mu_{mu}.tar')
    state_dict = torch.load(model_filename)
    return state_dict

def custom_hhc_eval(loader,model,device):

    model.eval()

    l_acc = []
    l_perf= []
    l_auc = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : testing HHCs'):
        bs,n,_ = target.shape
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        proba = torch.sigmoid()
        true_pos,n_total = accuracy_hhc(raw_scores,target)
        hhc_rec,total_hhcs = perf_hhc(raw_scores,target)
        l_acc.append((true_pos/n_total))
        l_perf.append((hhc_rec/total_hhcs))
        fpr, tpr, _ = skmetrics.roc_curve(target.cpu().detach().reshape(bs*n*n).numpy(), proba.cpu().detach().reshape(bs*n*n).numpy())
        l_auc.append(skmetrics.auc(fpr,tpr))
    return np.mean(l_acc),np.mean(l_perf),np.mean(l_auc)


if __name__=='__main__':
    gen_args = {
        'num_examples_train': 20000,
        'num_examples_val': 1000,
        'num_examples_test': 1000,
        'n_vertices': 50,
        'path_dataset': 'dataset_hhc',
        'generative_model': 'Gauss',
        'cycle_param': 0,
        'fill_param': None
    }
    opt_args = {
        'lr': 5e-5,
        'scheduler_step': 1,
        'scheduler_decay': 0.1
    }
    model_args = {
        'arch': 'simple', #siamese or simple
        'embedding': 'edge', #node or edge
        'num_blocks': 3,
        'original_features_num': 2,
        'in_features': 64,
        'out_features': 1,
        'depth_of_mlp': 3
    }
    helper_args = {
        'train':{# Training parameters
            'epoch': 10,
            'batch_size': 8,
            'lr': 1e-4,
            'scheduler_step': 1,
            'scheduler_decay': 0.5,
            'lr_stop': 1e-7,
            'print_freq': 100,
            'anew': True
        } 
    }

    n_vertices = gen_args['n_vertices']
    batch_size = helper_args['train']['batch_size']
    tot_epoch = helper_args['train']['epoch']
    pbm = 'hhc'

    start_param = 0
    end_param = 5
    steps = 20

    fill_param_list = np.sqrt(np.linspace(start_param,end_param**2,steps))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    path = 'hhc/'
    model_path = os.path.join(path,'models/')
    utils.check_dir(model_path)

    filename=f'hhc_results_{n_vertices}.txt'
    filepath = os.path.join(path,filename)
    n_lines=0
    if not os.path.isfile(filepath):
        with open(filepath,'w') as f:
            f.write('fill_param_train,acc,perf_hhc\n')
    else:
        with open(filepath,'r') as f:
            data = f.readlines()
            n_lines = len(data)-1
            print(f'File has {n_lines} computations')

    counter = 0
    pb = tqdm.tqdm(fill_param_list)
    for fill_param in pb:
        pb.set_description(desc=f'Using fill_param={fill_param}')

        if counter<n_lines:
            print(f'Skipping fill_param={fill_param}')
        else:
            if check_model_exists(model_path,n_vertices,fill_param): #If model already exists
                print(f'Using already trained model for mu={fill_param}')
                model = get_model(model_args)
                state_dict = load_model_dict(model_path,n_vertices,fill_param)
                model.load_state_dict(state_dict)
                model.to(device)
            else:
                gen_args['fill_param'] = fill_param
                train_gen = HHC_Generator('train',gen_args)
                train_gen.load_dataset()
                train_loader = siamese_loader(train_gen,batch_size,True,True)

                val_gen = HHC_Generator('val',gen_args)
                val_gen.load_dataset()
                val_loader = siamese_loader(val_gen,batch_size,True,True)

                helper = init_helper('hhc','train',helper_args)

                model = get_model(model_args)
                model.to(device)

                optimizer, scheduler = get_optimizer(opt_args,model)

                for epoch in range(tot_epoch):
                    train_triplet(train_loader,model,optimizer,helper,device,epoch,eval_score=True,print_freq=100)
                    
                    _, loss = val_triplet(val_loader,model,helper,device,epoch,eval_score=True)

                    scheduler.step(loss)

                    cur_lr = utils.get_lr(optimizer)
                    if helper.stop_condition(cur_lr):
                        print(f"Learning rate ({cur_lr}) under stopping threshold, ending training.")
                        break
                
                save_model(model_path, model, n_vertices, fill_param)

            test_gen = HHC_Generator('test',gen_args)
            test_gen.load_dataset()
            test_loader = siamese_loader(test_gen,batch_size,True,True)
            
            acc,hhc_proba,auc = custom_hhc_eval(test_loader,model,device)

            add_line(filepath,f'{fill_param},{acc},{hhc_proba},{auc}')

        counter+=1





