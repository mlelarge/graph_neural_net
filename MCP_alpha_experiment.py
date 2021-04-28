import os
from numpy.core.einsumfunc import _einsum_dispatcher
import torch
import numpy as np
import tqdm
from commander import get_model,load_model
from loaders.data_generator import MCP_Generator
from loaders.siamese_loaders import siamese_loader
from toolbox.searches import mcp_beam_method
import toolbox.utils as utils
from toolbox.optimizer import get_optimizer
from toolbox.helper import get_helper
from trainer import train_triplet, val_triplet

def add_line(filename,line) -> None:
    with open(filename,'a') as f:
        f.write(line + "\n")

def get_line(ptrain,ptest,error,accuracy):
    return f"{ptrain},{ptest},{error},{accuracy}"

def compute_a(n_vertices, edge_density):
    return -np.log(edge_density)/np.log(n_vertices)

def compute_cs(n_vertices,a):
    return 2/a+2*np.log(a)/(a*np.log(n_vertices))+2*np.log(np.exp(1)/2)/(a*np.log(n_vertices))+1

def init_helper(problem, name, config):
    exp_helper_object = get_helper(problem)
    exp_helper = exp_helper_object(name, config)
    return exp_helper

def custom_mcp_eval(loader,model,device):

    model.eval()

    l_errors = []
    l_acc = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : solving mcps'):
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        l_clique_inf = mcp_beam_method(data.squeeze(),raw_scores)
        l_clique_sol = utils.mcp_adj_to_ind(target)
        for inf,sol in zip(l_clique_inf,l_clique_sol):
            l_errors.append(len(sol)-len(inf))
            l_acc.append(len((inf.intersection(sol)))/len(sol))
    return l_errors,l_acc


if __name__=='__main__':
    gen_args = {
        'num_examples_train': 1000,
        'num_examples_val': 1000,
        'num_examples_test': 100,
        'n_vertices': 50,
        'planted': True,
        'path_dataset': 'dataset_mcp',
        'clique_size': None,
        'edge_density': None
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    filename = 'mcp_alpha_results.txt'
    n_lines=0
    if not os.path.isfile(filename):
        with open(filename,'w') as f:
            f.write('ptrain,ptest,error,accuracy\n')
    else:
        with open(filename,'r') as f:
            data = f.readlines()
            n_lines = len(data)-1
            print(f'File has {n_lines} computations')

    lp1 = np.arange(0.05,1,0.1)
    lp2 = np.arange(0.05,1,0.1)
    length = len(lp1)

    l_total = [(p1,p2) for p1 in lp1 for p2 in lp2]
    counter = 0
    max_cs1=0
    pb1 = tqdm.tqdm(lp1)
    for p1 in pb1:
        pb1.set_description(f"Outer Loop : p1={p1}\n")
        p1 = round(p1,2) #To prevent the 0.150000000002 as much as possible
        a1 = compute_a(n_vertices=n_vertices,edge_density=p1)
        cs1 = int(np.ceil(compute_cs(n_vertices,a1)))
        cs1 = max(max_cs1,cs1)
        max_cs1=cs1

        
        
        if counter+len(lp2)>n_lines:
            gen_args['clique_size'] = cs1
            gen_args['edge_density']= p1
            train_gen=MCP_Generator('train',gen_args)
            train_gen.load_dataset()
            train_loader = siamese_loader(train_gen,batch_size,True,shuffle=True)

            val_gen=MCP_Generator('val',gen_args)
            val_gen.load_dataset()
            val_loader = siamese_loader(val_gen, batch_size,True,shuffle=True)
            
            helper = init_helper(pbm,'train',helper_args)

            model = get_model(model_args)

            optimizer, scheduler = get_optimizer(opt_args,model)

            for epoch in range(tot_epoch):
                train_triplet(train_loader,model,optimizer,helper,device,epoch,eval_score=True,print_freq=100)
                
                relevant_metric, loss = val_triplet(val_loader,model,helper,device,epoch,eval_score=True)

                scheduler.step(loss)

                cur_lr = utils.get_lr(optimizer)
                if helper.stop_condition(cur_lr):
                    print(f"Learning rate ({cur_lr}) under stopping threshold, ending training.")
                    break
            #os.system(f"python3 commander.py train with data.train._mcp.clique_size={cs1} data.train._mcp.edge_density={p1}")
        
        max_cs2=0
        pb2 = tqdm.tqdm(lp2)
        for p2 in pb2:
            pb2.set_description(f'Inner Loop : p1={p1}, p2={p2}\n')
            p2 = round(p2,2)
            a2 = compute_a(n_vertices=n_vertices,edge_density=p2)
            cs2 = int(np.ceil(compute_cs(n_vertices,a2)))
            cs2 = max(max_cs2,cs2) #Keep growing cliques until we reset p2
            max_cs2=cs2
            if counter>=n_lines:
                gen_args['clique_size'] = cs2
                gen_args['edge_density']= p2
                gen_args=MCP_Generator('val',gen_args)
                gen_args.load_dataset()
                val_loader = siamese_loader(gen_args,batch_size,True,shuffle=True)

                l_errors,l_acc = custom_mcp_eval(val_loader,model,device)

                mean_error = np.mean(l_errors)
                mean_acc = np.mean(l_acc)
                line = get_line(p1,p2,mean_error,mean_acc)
                add_line(filename,line)
            else:
                print(f"Skipping ({p1},{p2})")
            counter+=1








