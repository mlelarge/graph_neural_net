from loaders.data_generator import SBM_Generator
import toolbox.utils as utils
import tqdm
import torch
import numpy as np
import os

from commander import get_model,init_helper
from toolbox.optimizer import get_optimizer
from loaders.siamese_loaders import siamese_loader
from trainer import train_triplet,val_triplet
from toolbox.metrics import accuracy_sbm_two_categories_edge

import sklearn.metrics as skmetrics

MODEL_NAME = 'model_cfixed-n_{}-dc_{}.tar'

def add_line(filename,line) -> None:
    with open(filename,'a') as f:
        f.write(line + '\n')

def check_model_exists(model_path,n_vertices,dc)->bool:
    model_filename = os.path.join(model_path,MODEL_NAME.format(n_vertices,dc))
    return os.path.isfile(model_filename)

def save_model(model_path,model,n_vertices,dc)->None:
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,MODEL_NAME.format(n_vertices,dc))
    torch.save(model.state_dict(), model_filename)

def load_model_dict(model_path,n_vertices,dc,device):
    utils.check_dir(model_path)
    model_filename = os.path.join(model_path,MODEL_NAME.format(n_vertices,dc))
    state_dict = torch.load(model_filename,map_location=device)
    return state_dict

def custom_sbm_eval(loader,model,device)->float:

    model.eval()

    l_acc = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : testing SBMs'):
        bs,n,_ = target.shape
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        true_pos,n_total = accuracy_sbm_two_categories_edge(raw_scores,target)
        l_acc.append((true_pos/n_total))
    acc = np.mean(l_acc)

    return acc


if __name__=='__main__':
    gen_args = {
        'num_examples_train': 10000,
        'num_examples_val': 1000,
        'num_examples_test': 1000,
        'n_vertices': 100,
        'path_dataset': 'dataset_sbm',
        'p_inter': None,
        'p_outer': None,
        'alpha': 0.5
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
            'epoch': 10,
            'batch_size': 4,
            'lr': 1e-5,
            'scheduler_step': 1,
            'scheduler_decay': 0.1,
            'lr_stop': 1e-7,
            'print_freq': 100,
            'anew': True
        } 
    }

    n_vertices = gen_args['n_vertices']
    batch_size = helper_args['train']['batch_size']
    tot_epoch = helper_args['train']['epoch']
    pbm = 'sbm'
    
    retrain = False
    n_retrain = 5
    if not retrain:
        n_retrain=1

    c=3

    start_dc = 0
    end_dc = 6
    steps = 20
    model_used = [-4]

    dc_list = np.linspace(start_dc,end_dc,steps)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    path = 'sbm/'
    model_path = os.path.join(path,'models/')
    utils.check_dir(model_path)

    filename_template='sbm_cfixed-dc_{}-' + f'n_{n_vertices}-c_{c}.txt'
    
    lname_template='sbm_cgen-l-dc_{}-'+ f'n_{n_vertices}-c_{c}.txt'


    
    pb = tqdm.tqdm(model_used)
    for model_idx in pb:
        dc = dc_list[model_idx]
        pb.set_description(desc=f'Using dc={dc}')
        lname = lname_template.format(dc)
        filename = filename_template.format(dc)
        p_inter = c-dc/2
        p_outer = c+dc/2
        gen_args['p_inter'] = p_inter
        gen_args['p_outer'] = p_outer

        
        filepath = os.path.join(path,filename)
        n_lines=0
        if not os.path.isfile(filepath):
            with open(filepath,'w') as f:
                f.write('dc,acc\n')
        else:
            with open(filepath,'r') as f:
                data = f.readlines()
                n_lines = len(data)-1
                print(f'File has {n_lines} computations')

        lpath = os.path.join(path,lname)
        l_n_lines=0
        if not os.path.isfile(lpath):
            with open(lpath,'w') as f:
                f.write('dc,list\n')
        else:
            with open(lpath,'r') as f:
                ldata = f.readlines()
                ln_lines = len(ldata)-1
                print(f'List file has {l_n_lines} computations')

        counter = 0
        lcounter= 0

        l_acc = []
        for _ in range(n_retrain):
            if not retrain and check_model_exists(model_path,n_vertices,dc): #If model already exists
                print(f'Using already trained model for dc={dc}')
                model = get_model(model_args)
                state_dict = load_model_dict(model_path,n_vertices,dc,device)
                model.load_state_dict(state_dict)
                model.to(device)
            else:
                train_gen = SBM_Generator('train',gen_args)
                train_gen.load_dataset()
                train_loader = siamese_loader(train_gen,batch_size,True,True)

                val_gen = SBM_Generator('val',gen_args)
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
                
                save_model(model_path, model, n_vertices, dc)

            pb2 = tqdm.tqdm(dc_list)
            for dc_test in pb2:
                pb2.set_description(f'Testing dc={dc} on dc_test={dc_test}')
                if counter<n_lines:
                    print(f'\nSkipping dc_test={dc_test}')
                else:
                    test_gen = SBM_Generator('test',gen_args)
                    test_gen.load_dataset()
                    test_loader = siamese_loader(test_gen,batch_size,True,True)
                    
                    cur_acc = custom_sbm_eval(test_loader,model,device)
                    l_acc.append(cur_acc)

        
        acc = np.mean(l_acc)
        add_line(filepath,f'{dc},{acc}')
        if retrain:
            add_line(lpath,f'{dc},{l_acc}')
        counter+=1