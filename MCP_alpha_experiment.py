import os
import torch
import numpy as np
import tqdm
from commander import ex,get_model,load_model
from loaders.data_generator import MCP_Generator
from loaders.siamese_loaders import siamese_loader
from toolbox.searches import mcp_beam_method
from toolbox.utils import mcp_adj_to_ind

def add_line(filename,line) -> None:
    with open(filename,'a') as f:
        f.write(line + "\n")

def get_line(ptrain,ptest,error,accuracy):
    return f"{ptrain},{ptest},{error},{accuracy}"

def compute_a(n_vertices, edge_density):
    return -np.log(edge_density)/np.log(n_vertices)

def compute_cs(n_vertices,a):
    return 2/a+2*np.log(a)/(a*np.log(n_vertices))+2*np.log(np.exp(1)/2)/(a*np.log(n_vertices))+1

@ex.command
def custom_mcp_train(p1,cs1,train_data_dict):
    train_data_dict['edge_density'] = p1
    train_data_dict['clique_size']  = cs1
    ex.add_config({'train_data_dict':train_data_dict})
    ex.command('train')

@ex.command
def custom_mcp_eval(p1,cs1,p2,cs2,cpu, test_data_dict, arch, train):

    test_data_dict['edge_density'] = p2
    test_data_dict['clique_size']  = cs2
    gen = MCP_Generator('test',test_data_dict)

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    model_path= f"./runs/Fusion/MCP-{p1}_{cs1}_True_dataset_mcp/model_best.pth.tar"

    model = get_model(arch)
    model.to(device)
    model = load_model(model, device,model_path=model_path)
    model.eval()

    gen.load_dataset()
    loader = siamese_loader(gen, train['batch_size'], gen.constant_n_vertices,shuffle=False)


    l_errors = []
    l_acc = []
    for data,target in tqdm.tqdm(loader,desc='Inner Loop : solving mcps'):
        data = data.to(device)
        target = target.to(device)
        raw_scores = model(data).squeeze(-1)
        l_clique_inf = mcp_beam_method(data.squeeze(),raw_scores)
        l_clique_sol = mcp_adj_to_ind(target)
        for inf,sol in zip(l_clique_inf,l_clique_sol):
            l_errors.append(len(sol)-len(inf))
            l_acc.append(len((inf.intersection(sol)))/len(sol))
    return l_errors,l_acc


if __name__=='__main__':
    

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

    n_vertices=50
    lp1 = np.arange(0.05,1,0.1)
    lp2 = np.arange(0.05,1,0.1)
    length = len(lp1)

    l_total = [(p1,p2) for p1 in lp1 for p2 in lp2]
    counter = 0
    max_cs1=0
    for p1,p2 in tqdm.tqdm(l_total):
        if counter%length==0:
            max_cs2 = 0
        p1 = round(p1,2) #To prevent the 0.150000000002 as much as possible
        p2 = round(p2,2)
        a1 = compute_a(n_vertices=n_vertices,edge_density=p1)
        a2 = compute_a(n_vertices=n_vertices,edge_density=p2)
        cs1 = int(np.ceil(compute_cs(n_vertices,a1)))
        cs1 = max(max_cs1,cs1)
        max_cs1=cs1
        cs2 = int(np.ceil(compute_cs(n_vertices,a2)))
        cs2 = max(max_cs2,cs2) #Keep growing cliques until we reset p2
        max_cs2=cs2
        if counter>=n_lines:
            os.system(f"python3 commander.py train with data.train._mcp.clique_size={cs1} data.train._mcp.edge_density={p1}")
            #os.system(f"python3 commander.py eval with data.test._mcp.clique_size={cs2} data.test._mcp.edge_density={p2}")
            ex.add_config({'p1':p1,'p2':p2,'cs1':cs1,'cs2':cs2})
            #ex.run('custom_mcp_train')
            l_errors,l_acc = ex.run('custom_mcp_eval').result
            mean_error = np.mean(l_errors)
            mean_acc = np.mean(l_acc)
            line = get_line(p1,p2,mean_error,mean_acc)
            add_line(filename,line)
        else:
            print(f"Skipping ({p1},{p2})")
        counter+=1








