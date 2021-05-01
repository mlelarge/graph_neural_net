from loaders.data_generator import MCP_Generator
from toolbox.searches import mcp_beam_method, mcp_proba_cheat
from numpy import log
import toolbox.utils as utils
import tqdm
import os
import torch
import numpy as np

from commander import get_model,init_helper
from toolbox.optimizer import get_optimizer
from loaders.siamese_loaders import siamese_loader
from trainer import train_triplet,val_triplet

import sklearn.metrics as skmetrics

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def approx_clique_size(n,p):
    return 2*log(n)/log(1/p)

def add_line(filename,line) -> None:
    with open(filename,'a') as f:
        f.write(line + '\n')

def get_line(cs1,cs2,cs_found,accuracy,auc):
    return f'{cs1},{cs2},{cs_found},{accuracy},{auc}'

def get_line_bl(cs1,cs_found,accuracy):
    return f'{cs1},{cs_found},{accuracy}'

def custom_mcp_eval(loader,model,device):

    model.eval()

    l_inf_cs = []
    l_acc = []
    l_auc = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : solving mcps'):
        bs,n,_ = target.shape
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        proba = torch.sigmoid(raw_scores)
        l_clique_inf = mcp_beam_method(data.squeeze(),raw_scores,beam_size=500)
        l_clique_sol = utils.mcp_adj_to_ind(target)
        for inf,sol in zip(l_clique_inf,l_clique_sol):
            l_inf_cs.append(len(inf))
            l_acc.append(len((inf.intersection(sol)))/len(sol))
        
        fpr, tpr, _ = skmetrics.roc_curve(target.cpu().detach().reshape(bs*n*n).numpy(), proba.cpu().detach().reshape(bs*n*n).numpy())
        l_auc.append(skmetrics.auc(fpr,tpr))

    return l_inf_cs,l_acc,l_auc

def custom_mcp_bl_eval(loader):

    l_bl_cs = []
    l_acc = []
    for data,target in tqdm.tqdm(loader,desc='Inner baseline loop : solving mcps'):
        bs,n,_ = target.shape
        data_adj = data[:,:,:,1]
        l_clique_inf = mcp_beam_method(data.squeeze(),data_adj,beam_size=500)
        l_clique_sol = utils.mcp_adj_to_ind(target)
        for inf,sol in zip(l_clique_inf,l_clique_sol):
            l_bl_cs.append(len(inf))
            l_acc.append(len((inf.intersection(sol)))/len(sol))
        

    return l_bl_cs,l_acc,l_auc

if __name__=='__main__':
    gen_args = {
        'num_examples_train': 20000,
        'num_examples_val': 1000,
        'num_examples_test': 20,
        'n_vertices': 50,
        'planted': True,
        'path_dataset': 'dataset_mcp',
        'clique_size': None,
        'edge_density': 0.5
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

    tot_epoch=helper_args['train']['epoch']
    batch_size = helper_args['train']['batch_size']
    n_vertices=gen_args['n_vertices']
    pbm='mcp'
    
    cs_start=7
    cs_stop=21
    l_cs = range(cs_start,cs_stop)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    filename='mcp_pc_results.txt'
    n_lines=0
    if not os.path.isfile(filename):
        with open(filename,'w') as f:
            f.write('cs_train,cs_test,inf_cs,bl_cs,acc,auc\n')
    else:
        with open(filename,'r') as f:
            data = f.readlines()
            n_lines = len(data)-1
            print(f'File has {n_lines} computations')

    bl_n_lines = 0
    baseline_filename='mcp_ps_baseline.txt'
    if not os.path.isfile(baseline_filename):
        with open(baseline_filename,'w') as f:
            f.write('cs,cs_inferred,acc\n')
    else:
        with open(baseline_filename,'r') as f:
            data = f.readlines()
            bl_n_lines = len(data)-1
            print(f'Baseline file has {bl_n_lines} computations')


    counter = 0
    bl_counter=0
    pbcs = tqdm.tqdm(l_cs)
    for cs1 in pbcs:
        if counter+len(l_cs)<=n_lines:
            print(f'\nSkipping model for cs1={cs1}')
        else:
            pbcs.set_description(f'Using training clique_size {cs1}')

            gen_args['clique_size'] = cs1
            train_gen=MCP_Generator('train',gen_args)
            train_gen.load_dataset()
            
            l_real_cs = [utils.get_cs(t) for _,t in train_gen.data]
            
            kwargs = {'alpha':.6,'fc':'dodgerblue','ec':'navy'}
            plt.figure()
            plt.hist(l_real_cs,bins=range(cs_start,cs_stop+1),density=True,align='left',**kwargs)
            plt.title(f"Density of real clique size for planting size {cs1}")
            plt.xlabel("Clique Size")
            plt.tight_layout()
            plt.xticks(l_cs)
            plt.xlim(l_cs[0]-1,l_cs[-1]+1)
            plt.show()
            plt.savefig(f'figures/mcp_ps/displot-cs_{cs1}')

            train_loader = siamese_loader(train_gen,batch_size,True,shuffle=True)

            val_gen=MCP_Generator('val',gen_args)
            val_gen.load_dataset()
            val_loader = siamese_loader(val_gen, batch_size,True,shuffle=True)
            
            helper = init_helper(pbm,'train',helper_args)

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
            
            pb2 = tqdm.tqdm(l_cs)
            for cs2 in pb2:
                if counter<n_lines:
                    print(f'\nSkipping cs1={cs1}, cs2={cs2}')
                    continue
                
                pb2.set_description(f'cs1={cs1}, cs2={cs2}')
                
                gen_args['clique_size'] = cs2
                test_gen=MCP_Generator('test',gen_args)
                test_gen.load_dataset()
                test_loader = siamese_loader(test_gen,batch_size,True,shuffle=True)

                l_cs_found,l_acc,l_auc = custom_mcp_eval(test_loader,model,device)

                mean_cs_found = np.mean(l_cs_found)
                mean_acc = np.mean(l_acc)
                mean_auc = np.mean(l_auc)
                line = get_line(cs1,cs2,mean_cs_found,mean_acc,mean_auc)
                add_line(filename,line)
        if bl_n_lines>counter: #If we've already computed the baseline values, next iteration
            print(f'\nSkipping baseline for {cs1}')
        else:
            gen_args['clique_size'] = cs1
            test_gen=MCP_Generator('test',gen_args)
            test_gen.load_dataset()
            test_loader = siamese_loader(test_gen,batch_size,True,shuffle=True)
    
            l_bl_cs,l_bl_acc = custom_mcp_bl_eval(test_loader)
            mean_cs_bl = np.mean(l_cs_found)
            mean_acc = np.mean(l_acc)
            line = get_line_bl(cs1,mean_cs_bl,mean_acc)
            add_line(baseline_filename,line)

        counter+=1
        
        

