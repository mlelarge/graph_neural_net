from loaders.data_generator import MCP_Generator,MCP_True_Generator
from toolbox.searches import mcp_beam_method
from numpy import log
import toolbox.utils as utils
import tqdm
import os
import torch
import numpy as np
import pandas as pd

from commander import get_model,init_helper
from toolbox.optimizer import get_optimizer
from loaders.siamese_loaders import siamese_loader
from trainer import train_triplet,val_triplet

import sklearn.metrics as skmetrics

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def approx_clique_size(n,p)->float:
    return 2*log(n)/log(1/p)

def add_line(filename,line) -> None:
    with open(filename,'a') as f:
        f.write(line + '\n')

def get_line(cs1,cs2,cs_found,accuracy,auc)->str:
    return f'{cs1},{cs2},{cs_found},{accuracy},{auc}'

def get_line_bl(cs1,cs_found,accuracy)->str:
    return f'{cs1},{cs_found},{accuracy}'

def check_model_exists(model_path,n_vertices,cs)->bool:
    model_filename = os.path.join(model_path,f'model-n_{n_vertices}-cs_{cs}.tar')
    return os.path.isfile(model_filename)

def save_model(model_path,model,n_vertices,cs)->None:
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,f'model-n_{n_vertices}-cs_{cs}.tar')
    torch.save(model.state_dict(), model_filename)

def load_model_dict(model_path,n_vertices,cs,device):
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,f'model-n_{n_vertices}-cs_{cs}.tar')
    state_dict = torch.load(model_filename,map_location=device)
    return state_dict

def custom_mcp_vanilla_eval(loader,model,device):

    model.eval()

    l_inf_cs = []
    for data,target in tqdm.tqdm(loader,desc='Inner loop : solving mcps'):
        bs,n,_ = target.shape
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        l_clique_inf = mcp_beam_method(data.squeeze(),raw_scores,beam_size=500)
        for inf in l_clique_inf:
            l_inf_cs.append(len(inf))

    return l_inf_cs

def custom_mcp_bl_eval(loader):

    l_bl_cs = []
    l_acc = []
    for data,target in tqdm.tqdm(loader,desc='Inner baseline loop : solving mcps'):
        bs,n,_ = target.shape
        data_adj = data[:,:,:,1]
        l_clique_inf = mcp_beam_method(data.squeeze(),data_adj,beam_size=500)
        for inf in l_clique_inf:
            l_bl_cs.append(len(inf))
        

    return l_bl_cs,l_acc

if __name__=='__main__':
    gen_args = {
        'num_examples_train': 20000,
        'num_examples_val': 1000,
        'num_examples_test': 100,
        'n_vertices': 50,
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

    path = 'mcp_ps/'
    model_path = os.path.join(path,'models/')
    utils.check_dir(model_path)

    tot_epoch=helper_args['train']['epoch']
    batch_size = helper_args['train']['batch_size']
    n_vertices=gen_args['n_vertices']
    pbm='mcp'
    
    cs_start=7
    cs_stop=21
    l_cs = range(cs_start,cs_stop)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    filename=f'mcp_v_results-{n_vertices}.txt'
    filepath = os.path.join(path,filename)
    n_lines=0
    if not os.path.isfile(filepath):
        with open(filepath,'w') as f:
            f.write('cs_train,cs_inf\n')
    else:
        with open(filepath,'r') as f:
            data = f.readlines()
            n_lines = len(data)-1
            print(f'File has {n_lines} computations')

    bl_n_lines = 0
    baseline_filename=f'mcp_v_baseline-{n_vertices}.txt'
    bl_path = os.path.join(path,baseline_filename)
    if not os.path.isfile(bl_path):
        with open(bl_path,'w') as f:
            f.write('cs_test,cs_inf\n')
    else:
        with open(bl_path,'r') as f:
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

            
            if check_model_exists(model_path,n_vertices,cs1): #If model already exists
                print(f'Using already trained model for cs1={cs1}')
                model = get_model(model_args)
                state_dict = load_model_dict(model_path,n_vertices,cs1,device)
                model.load_state_dict(state_dict)
                model.to(device)
            else: #If model doesn't exist, we need to train
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
                
                save_model(model_path,model,n_vertices,cs1)
            
        if n_lines>counter:
            print(f'\nSkipping model test for {cs1}')
        else:
            gen_args['clique_size'] = cs1
            test_gen=MCP_True_Generator('test',gen_args)
            test_gen.load_dataset()
            test_loader = siamese_loader(test_gen,batch_size,True,shuffle=True)
    
            l_bl_cs = custom_mcp_vanilla_eval(test_loader)
            mean_cs_bl = np.mean(l_bl_cs)
            line = get_line(cs1,mean_cs_bl)
            add_line(bl_path,line)

        if bl_n_lines>counter: #If we've already computed the baseline values, next iteration
            print(f'\nSkipping baseline for {cs1}')
        else:
            gen_args['clique_size'] = cs1
            test_gen=MCP_True_Generator('test',gen_args)
            test_gen.load_dataset()
            test_loader = siamese_loader(test_gen,batch_size,True,shuffle=True)
    
            l_bl_cs,l_bl_acc = custom_mcp_bl_eval(test_loader)
            mean_cs_bl = np.mean(l_bl_cs)
            mean_acc = np.mean(l_bl_acc)
            line = get_line_bl(cs1,mean_cs_bl)
            add_line(bl_path,line)

        counter+=1
    
    model_data = pd.read_csv(filepath,delimiter=',')
    bl_data = pd.read_csv(bl_path,delimiter=',')

    model_data['bl_perf']=0
    for row_n,row in bl_data.iterrows():
        cs_test = row['cs_test']
        cs_inf = row['cs_inf']
        model_data.loc[model_data['cs_test']==cs_test, 'bl_perf'] = cs_inf
    
    imax = model_data['cs_train'].max()
    imin = model_data['cs_test'].min()
    heatmap_size = imax-imin+1
    model_data['i'] = imax-model_data['cs_train']
    model_data['j'] = model_data['cs_test']-imin

    heatmap = np.zeros((heatmap_size,heatmap_size))
    for _,row in model_data.iterrows():
        i,j = int(row['i']),int(row['j'])
        heatmap[i,j] = row['auc']#row['cs_inf']-row['bl_perf']


    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    fig.colorbar(im,ax=ax)
    ax.set_xticks(range(heatmap_size))
    ax.set_xticklabels(range(imin,imax+1))
    ax.set_yticks(range(heatmap_size))
    ax.set_yticklabels(range(imin,imax+1)[::-1])
    ax.set_xlabel("Testing clique size")
    ax.set_ylabel('Training clique size')
    for i in range(heatmap_size):
        for j in range(heatmap_size):
            text = ax.text(j, i, round(heatmap[i, j],2),
                        ha="center", va="center", color="b")
    fig.tight_layout()
    plt.show()
    plt.savefig('mcp_planted.png',format='PNG')
    

