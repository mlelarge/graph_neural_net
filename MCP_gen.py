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
from toolbox.metrics import accuracy_inf_sol_multiple, accuracy_mcp, accuracy_mcp_exact
from toolbox.searches import mc_bronk2_cpp,mc_bronk2, mcp_beam_method

import sklearn.metrics as skmetrics

MODEL_NAME = 'model-n_{}-cs_{}.tar'

def add_line_lists(filename,param1,values_list):
    chaine = f'{param1}'
    assert len(values_list)>0, "Empty list, can't write anything."
    try:
        n_to_write = len(values_list[0])
    except TypeError:
        n_to_write=1
        values_list = [[elt] for elt in values_list]
    for value_number in range(n_to_write):
        chaine+=','
        l_temp_values = [elt[value_number] for elt in values_list]
        chaine+=str(l_temp_values)
    add_line(filename,chaine)

def add_line_mean(filename,param1,values_list):
    chaine = f'{param1}'
    assert len(values_list)>0, "Empty list, can't write anything."
    try:
        n_to_write = len(values_list[0])
    except TypeError:
        n_to_write=1
        values_list = [[elt] for elt in values_list]
    for value_number in range(n_to_write):
        chaine+=','
        l_temp_values = [elt[value_number] for elt in values_list]
        chaine+=str(np.mean(l_temp_values))
    add_line(filename,chaine)

def add_line_mean(filename,param1,values_list):
    chaine = f'{param1}'
    for i in range(len(values_list[0])):
        value = np.mean([it_value[i] for it_value in values_list])
        chaine+= ',' + str(value)
    add_line(filename,chaine)

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

def custom_mcp_eval(loader,model,device):

    model.eval()

    l_acc_inf_mcps = []
    l_acc_mcps_mcp = []
    l_acc_inf_mcp  = []
    l_auc_inf_mcps = []
    l_auc_inf_mcp  = []
    l_cs_inf_mcp = []
    l_cs_mcps_mcp= []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : solving MCPs'):
        bs,n,_ = target.shape
        target_mcp = mc_bronk2_cpp(data[:,:,:,1].to(int))
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        probas = torch.sigmoid(raw_scores)

        inf_cliques = mcp_beam_method(data,raw_scores,beam_size=500)
        cs_inf_cliques = np.array([len(elt) for elt in inf_cliques])
        target_as_set = [utils.mcp_adj_to_ind(elt) for elt in target]
        cs_target_cliques = np.array([len(elt) for elt in target_as_set])
        cs_sol_cliques = np.array([len(elt[0]) for elt in target_mcp])

        l_cs_inf_mcp.append(np.sum(cs_inf_cliques/cs_sol_cliques))
        l_cs_mcps_mcp.append(np.sum(cs_target_cliques/cs_sol_cliques))
        

        tp,ntot,_ = accuracy_inf_sol_multiple(inf_cliques, [[elt] for elt in target_as_set])
        l_acc_inf_mcps.append(tp/ntot)
        tp,ntot,_ = accuracy_inf_sol_multiple(target_as_set, target_mcp)
        l_acc_mcps_mcp.append(tp/ntot)
        tp,ntot,cliques_inf_mcp = accuracy_inf_sol_multiple(inf_cliques, target_mcp)
        l_acc_inf_mcp.append(tp/ntot)
        
        target_inf_mcp = torch.zeros((bs,n,n))
        for k in range(bs):
            cur_clique = cliques_inf_mcp[k]
            target_inf_mcp[k] = utils.ind_to_adj(cur_clique,n).detach().clone()

        fpr, tpr, _ = skmetrics.roc_curve(target.cpu().detach().reshape(bs*n*n).numpy(), probas.cpu().detach().reshape(bs*n*n).numpy())
        l_auc_inf_mcps.append(skmetrics.auc(fpr,tpr))


        fpr, tpr, _ = skmetrics.roc_curve(target_inf_mcp.cpu().detach().reshape(bs*n*n).numpy(), probas.cpu().detach().reshape(bs*n*n).numpy())
        l_auc_inf_mcp.append(skmetrics.auc(fpr,tpr))
    acc_inf_mcps = np.mean(l_acc_inf_mcps)
    acc_mcps_mcp = np.mean(l_acc_mcps_mcp)
    acc_inf_mcp  = np.mean(l_acc_inf_mcp)
    auc_inf_mcps = np.mean(l_auc_inf_mcps)
    auc_inf_mcp  = np.mean(l_auc_inf_mcp)
    cs_inf_mcp   = np.mean(l_cs_inf_mcp)
    cs_mcps_mcp  = np.mean(l_cs_mcps_mcp)
    return acc_inf_mcps,acc_mcps_mcp,acc_inf_mcp,auc_inf_mcps,auc_inf_mcp,cs_inf_mcp,cs_mcps_mcp


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
    pbm = 'mcp'

    retrain=False
    n_retrain = 5
    if not retrain:
        n_retrain=1

    start_cs = 5
    end_cs = 20
    cs_model_used = [10,9]

    cs_list = range(start_cs,end_cs+1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    path = 'mcp/'
    model_path = os.path.join(path,'models/')
    utils.check_dir(model_path)

    filename_template='mcp_gen-cs_{}-'+f'n_{n_vertices}.txt'
    
    lname_template='mcp_gen-cs{}-'+f'n_{n_vertices}.txt'


    
    pb = tqdm.tqdm(cs_model_used)
    for cs in pb:
        pb.set_description(desc=f'Using cs={cs}')
        lname = lname_template.format(cs)
        filename = filename_template.format(cs)
        gen_args['clique_size'] = cs

        
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

        lpath = os.path.join(path,lname)
        l_n_lines=0
        if not os.path.isfile(lpath):
            with open(lpath,'w') as f:
                f.write('cs,acc_inf_mcps,acc_mcps_mcp,acc_inf_mcp,auc_inf_mcps,auc_inf_mcp\n')
        else:
            with open(lpath,'r') as f:
                ldata = f.readlines()
                ln_lines = len(ldata)-1
                print(f'List file has {l_n_lines} computations')

        counter = 0
        lcounter= 0
        l_l_values = [list() for _ in cs_list]
        for _ in range(n_retrain):
            if not retrain and check_model_exists(model_path,n_vertices,cs): #If model already exists
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
            pb2 = tqdm.trange(len(cs_list))
            counter=0
            for i in pb2:
                cs_test = cs_list[i]
                pb2.set_description(f'Testing cs={cs} on cs_test={cs_test}')
                if counter<n_lines:
                    print(f'\nSkipping cs_test={cs_test}')
                else:
                    l_values = []
                    gen_args['clique_size'] = cs_test

                    test_gen = MCP_Generator('test',gen_args)
                    test_gen.load_dataset()
                    test_loader = siamese_loader(test_gen,batch_size,True,True)
                    
                    l_l_values[i].append(custom_mcp_eval(test_loader,model,device))
                    #acc_inf_mcps,acc_mcps_mcp,acc_inf_mcp,auc_inf_mcps,auc_inf_mcp = custom_mcp_eval(test_loader,model,device)
                    #add_line_mean(filepath,cs_test,l_values)
                    #add_line_lists(lpath,cs_test,l_values)
                    #add_line(filepath,f'{cs},{acc_inf_mcps},{acc_mcps_mcp},{acc_inf_mcp},{auc_inf_mcps},{auc_inf_mcp}')

                counter+=1
            
        for i,l_values in enumerate(l_l_values):
            add_line_mean(filepath,cs_list[i],l_values)
            add_line_lists(lpath,cs_list[i],l_values)




