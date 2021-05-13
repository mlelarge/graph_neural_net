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
from toolbox.metrics import accuracy_mcp


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

    l_acc = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : testing SBMs'):
        bs,n,_ = target.shape
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        true_pos,n_total = accuracy_mcp(raw_scores,target)
        l_acc.append((true_pos/n_total))
    acc = np.mean(l_acc)
    return acc


if __name__=='__main__':
    gen_args = {
        'num_examples_train': 20000,
        'num_examples_val': 1000,
        'num_examples_test': 1000,
        'n_vertices': 100,
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

    filename=f'mcp_f1-n_{n_vertices}.txt'
    filepath = os.path.join(path,filename)
    n_lines=0
    if not os.path.isfile(filepath):
        with open(filepath,'w') as f:
            f.write('cs,acc\n')
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
                
                save_model(model_path, model, n_vertices, c)

            test_gen = MCP_Generator('test',gen_args)
            test_gen.load_dataset()
            test_loader = siamese_loader(test_gen,batch_size,True,True)
            
            acc = custom_mcp_eval(test_loader,model,device)

            add_line(filepath,f'{cs},{acc}')

        counter+=1





