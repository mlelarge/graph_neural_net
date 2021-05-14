from loaders.data_generator import MCP_Generator,MCP_True_Generator
import toolbox.utils as utils
import tqdm
import torch
import numpy as np
import os

from commander import get_model,init_helper
from toolbox.optimizer import get_optimizer
from loaders.siamese_loaders import siamese_loader
from trainer import train_triplet,val_triplet
from toolbox.metrics import accuracy_mcp, accuracy_mcp_exact
from toolbox.searches import mc_bronk2

import sklearn.metrics as skmetrics

MODEL_NAME = 'model-n_{}-cs_{}.tar'

def add_line(filename,line) -> None:
    with open(filename,'a') as f:
        f.write(line + '\n')

def check_model_exists(model_path,n_vertices,pvalue)->bool:
    model_filename = os.path.join(model_path,MODEL_NAME.format(n_vertices,pvalue))
    return os.path.isfile(model_filename)

def save_model(model_path,model,n_vertices,pvalue)->None:
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,MODEL_NAME.format(n_vertices,pvalue))
    torch.save(model.state_dict(), model_filename)

def load_model_dict(model_path,n_vertices,pvalue,device):
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,MODEL_NAME.format(n_vertices,pvalue))
    state_dict = torch.load(model_filename,map_location=device)
    return state_dict

def custom_mcp_eval(loader,model,device)->float:

    model.eval()

    l_acc_inf_mcps = []
    l_acc_mcps_mcp = []
    l_acc_inf_mcp  = []
    l_auc_inf_mcps = []
    l_auc_inf_mcp  = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : solving MCPs'):
        bs,n,_ = target.shape
        target_mcp =[]
        for k in range(bs):
            cliques_sol = mc_bronk2(data[:,:,:,1].to(int))
            target_mcp.append(cliques_sol)
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        probas = torch.sigmoid(raw_scores)

        tp,ntot = accuracy_mcp(raw_scores, target)
        l_acc_inf_mcps.append(tp/ntot)
        tp,ntot,cliques_inf_mcp = accuracy_mcp_exact(target, target_mcp)
        l_acc_mcps_mcp.append(tp/ntot)
        tp,ntot,_ = accuracy_mcp_exact(raw_scores, target_mcp)
        l_acc_inf_mcp.append(tp/ntot)
        
        target_inf_mcp = torch.zeros((bs,n,n))
        for k in range(bs):
            cur_clique = cliques_inf_mcp[k]
            target_inf_mcp[k] = utils.mcp_ind_to_adj(cur_clique,n).detach().clone()

        fpr, tpr, _ = skmetrics.roc_curve(target.cpu().detach().reshape(bs*n*n).numpy(), probas.cpu().detach().reshape(bs*n*n).numpy())
        l_auc_inf_mcps.append(skmetrics.auc(fpr,tpr))


        fpr, tpr, _ = skmetrics.roc_curve(target_inf_mcp.cpu().detach().reshape(bs*n*n).numpy(), probas.cpu().detach().reshape(bs*n*n).numpy())
        l_auc_inf_mcp.append(skmetrics.auc(fpr,tpr))
    acc_inf_mcps = np.mean(l_acc_inf_mcps)
    acc_mcps_mcp = np.mean(l_acc_mcps_mcp)
    acc_inf_mcp  = np.mean(l_acc_inf_mcp)
    auc_inf_mcps = np.mean(l_auc_inf_mcps)
    auc_inf_mcp  = np.mean(l_auc_inf_mcp)
    return acc_inf_mcps,acc_mcps_mcp,acc_inf_mcp,auc_inf_mcps,auc_inf_mcp


if __name__=='__main__':
    gen_args = {
        'num_examples_train': 10000,
        'num_examples_val': 1000,
        'num_examples_test': 100,
        'n_vertices': 100,
        'planted': True,
        'path_dataset': 'dataset_mcp',
        'clique_size': None,
        'edge_density': 0.5
    }
    opt_args = {
        'lr': 1e-5,
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
            'epoch': 20,
            'batch_size': 4,
            'lr': 1e-5,
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
    pbm = 'sbm'

    

    start_cs = 5
    end_cs = 15

    cs_list = range(start_cs,end_cs+1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    path = 'mcp/'
    model_path = os.path.join(path,'models/')
    utils.check_dir(model_path)

    filename=f'mcp_comp-n_{n_vertices}.txt'
    filepath = os.path.join(path,filename)
    n_lines=0
    if not os.path.isfile(filepath):
        with open(filepath,'w') as f:
            f.write('cs,acc_inf_mcps,acc_mcps_mcp,acc_inf_mcp,auc_inf_mcps,auc_inf_mcp\n')
    else:
        with open(filepath,'r') as f:
            data = f.readlines()
            n_lines = len(data)-1
            print(f'File has {n_lines} computations')

    counter = 0
    pb = tqdm.tqdm(cs_list)
    for cs in pb:
        pb.set_description(desc=f'Using cs={cs}')
        gen_args['clique_size'] = cs

        if counter<n_lines:
            print(f'Skipping cs={cs}')
        else:
            if check_model_exists(model_path,n_vertices,cs): #If model already exists
                print(f'Using already trained model for cs={cs}')
                model = get_model(model_args)
                state_dict = load_model_dict(model_path,n_vertices,cs,device)
                model.load_state_dict(state_dict)
                model.to(device)
            else:
                train_gen = MCP_Generator('train',gen_args)
                train_gen.load_dataset()
                train_loader = siamese_loader(train_gen,batch_size,True,True)

                val_gen = MCP_Generator('val',gen_args)
                val_gen.load_dataset()
                val_loader = siamese_loader(val_gen,batch_size,True,True)

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
                
                save_model(model_path, model, n_vertices, cs)

            test_gen = MCP_True_Generator('test',gen_args)
            test_gen.load_dataset()
            test_loader = siamese_loader(test_gen,batch_size,True,True)
            
            acc_inf_mcps,acc_mcps_mcp,acc_inf_mcp,auc_inf_mcps,auc_inf_mcp = custom_mcp_eval(test_loader,model,device)

            add_line(filepath,f'{cs},{acc_inf_mcps},{acc_mcps_mcp},{acc_inf_mcp},{auc_inf_mcps},{auc_inf_mcp}')

        counter+=1





