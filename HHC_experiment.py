from loaders.data_generator import HHCTSP_Generator
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
from toolbox.searches import tsp_beam_decode

pyconcorde_available=True
try:
    from concorde.tsp import TSPSolver
except ModuleNotFoundError:
    pyconcorde_available=False
    print("Trying to continue without pyconcorde as it is not installed. TSP data generation will fail.")

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

    l_tsp_len = []
    l_inf_len = []
    l_tspstar_len = []
    l_tsps_tsp_ratio = []
    l_tsps_tsp_edge_ratio = []
    l_inf_tsp_ratio = []
    l_inf_tsp_edge_ratio = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : testing HHCs'):
        bs,n,_ = target.shape
        hhc_target = torch.eye(n).roll(1,dims=-1)
        hhc_target = (hhc_target+hhc_target.T)
        hhc_target = hhc_target.unsqueeze(0)
        hhc_target = hhc_target.repeat( (bs,1,1) )
        hhc_target.to(device)
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        proba = torch.sigmoid(raw_scores)
        true_pos,n_total = accuracy_hhc(raw_scores,hhc_target)
        hhc_rec,total_hhcs = perf_hhc(raw_scores,hhc_target)
        l_acc.append((true_pos/n_total))
        l_perf.append((hhc_rec/total_hhcs))
        fpr, tpr, _ = skmetrics.roc_curve(target.cpu().detach().reshape(bs*n*n).numpy(), proba.cpu().detach().reshape(bs*n*n).numpy())
        l_auc.append(skmetrics.auc(fpr,tpr))

        W_dists = data[:,:,:,1]
        inf_tours = tsp_beam_decode(raw_scores,W_dists=W_dists)
        inf_len  = torch.sum(W_dists*inf_tours ,axis=-1).sum(axis=-1)
        tsp_len  = torch.sum(W_dists*target    ,axis=-1).sum(axis=-1)
        tsps_len = torch.sum(W_dists*hhc_target,axis=-1).sum(axis=-1)
        
        number_of_edges = 2*n
        l_tsps_tsp_edge_ratio.append(torch.sum(hhc_target*target).item()/(bs*number_of_edges))
        l_inf_tsp_edge_ratio.append(torch.sum(inf_tours *target).item()/(bs*number_of_edges))

        l_tsp_len.append(torch.sum(tsp_len).item())
        l_inf_len.append(torch.sum(inf_len).item())
        l_tspstar_len.append(torch.sum(tsps_len).item())

        l_inf_tsp_ratio.append(torch.sum(inf_len/tsp_len).item())
        l_tsps_tsp_ratio.append(torch.sum(tsps_len/tsp_len).item())
    acc = np.mean(l_acc)
    perf = np.mean(l_perf)
    auc = np.mean(auc)
    tsp_len = np.mean(l_tsp_len)
    inf_len = np.mean(l_inf_len)
    tsps_len = np.mean(l_tspstar_len)
    tsps_tsp_ratio = np.mean(l_tsps_tsp_ratio)
    tsps_tsp_edge_ratio  = np.mean(l_tsps_tsp_edge_ratio)
    inf_tsp_ratio = np.mean(l_inf_tsp_ratio)
    inf_tsp_edge_ratio = np.mean(l_inf_tsp_edge_ratio)

    return acc,perf,auc,tsp_len,inf_len,tsps_len,tsps_tsp_ratio,tsps_tsp_edge_ratio,inf_tsp_ratio,inf_tsp_edge_ratio


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
            f.write('fill_param_train,acc,perf_hhc,auc,tsp_length,tsps_length,inf_length,tsps_tsp_len_ratio,tsps_tsp_edge_ratio,inf_tsp_len_ratio,inf_tsp_edge_ratio\n')
    else:
        with open(filepath,'r') as f:
            data = f.readlines()
            n_lines = len(data)-1
            print(f'File has {n_lines} computations')

    counter = 0
    pb = tqdm.tqdm(fill_param_list)
    for fill_param in pb:
        pb.set_description(desc=f'Using fill_param={fill_param}')
        gen_args['fill_param'] = fill_param

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
                train_gen = HHCTSP_Generator('train',gen_args)
                train_gen.load_dataset()
                train_loader = siamese_loader(train_gen,batch_size,True,True)

                val_gen = HHCTSP_Generator('val',gen_args)
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

            test_gen = HHCTSP_Generator('test',gen_args)
            test_gen.load_dataset()
            test_loader = siamese_loader(test_gen,batch_size,True,True)
            
            acc,hhc_proba,auc, tsp_len,inf_len,tspstar_len,tsp_tsps_ratio,tsps_tsp_edge_ratio,inf_tsp_ratio,inf_tsp_edge_ratio = custom_hhc_eval(test_loader,model,device)

            add_line(filepath,f'{fill_param},{acc},{hhc_proba},{auc},{tsp_len},{inf_len},{tspstar_len},{tsp_tsps_ratio},{tsps_tsp_edge_ratio},{inf_tsp_ratio},{inf_tsp_edge_ratio}')

        counter+=1





