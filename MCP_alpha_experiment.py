from re import search
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

def get_line(ptrain,ptest,error):
    return f"{ptrain},{ptest},{error}"

def compute_a(n_vertices, edge_density):
    return -np.log(edge_density)/np.log(n_vertices)

def compute_cs(n_vertices,a):
    return 2/a+2*np.log(a)/(a*np.log(n_vertices))+2*np.log(np.exp(1)/2)/(a*np.log(n_vertices))+1

@ex.command
def custom_mcp_train(p1,cs1,train_data_dict):
    train_data_dict['edge_density'] = p1
    train_data_dict['clique_size']  = cs1
    ex.command('train',config_updates={
        'train_data_dict': train_data_dict
        }
    )

@ex.command
def custom_mcp_eval(p2,cs2,cpu, test_data_dict, arch):
    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    model = get_model(arch)
    model.to(device)
    model = load_model(model, device)
    model.eval()


    test_data_dict['edge_density'] = p2
    test_data_dict['clique_size']  = cs2
    gen = MCP_Generator('test',test_data_dict)
    gen.load_dataset()
    loader = siamese_loader(gen, 1, gen.constant_n_vertices,shuffle=False)


    l_errors = []
    for data,target in loader:
        raw_scores = model(data)
        l_clique_inf = mcp_beam_method(data,raw_scores)
        l_clique_sol = mcp_adj_to_ind(target)
        for inf,sol in zip(l_clique_inf,l_clique_sol):
            l_errors.append(len(sol)-len(inf))
    return l_errors


if __name__=='__main__':
    
    filename = 'mcp_alpha_results.txt'
    with open(filename,'w') as f:
        f.write('ptrain,ptest,error\n')


    n_vertices=50
    lp1 = np.arange(0.05,1,0.1)
    lp2 = np.arange(0.05,1,0.1)

    l_total = [(p1,p2) for p1 in lp1 for p2 in lp2]
    for p1,p2 in tqdm.tqdm(l_total):
        p1 = round(p1,2) #To prevent the 0.150000000002 as much as possible
        p2 = round(p2,2)
        a1 = compute_a(n_vertices=n_vertices,edge_density=p1)
        a2 = compute_a(n_vertices=n_vertices,edge_density=p2)
        cs1 = int(np.ceil(compute_cs(n_vertices,a1)))
        cs2 = int(np.ceil(compute_cs(n_vertices,a2)))
        #os.system(f"python3 commander.py train with data.train._mcp.clique_size={cs1} data.train._mcp.edge_density={p1}")
        #os.system(f"python3 commander.py eval with data.test._mcp.clique_size={cs2} data.test._mcp.edge_density={p2}")
        ex.run("train",config_updates={
            'data.train._mcp.clique_size':cs1,
            'data.train._mcp.edge_density':p1
            })
        l_errors = ex.run('custom_mcp',config_updates={
            'data.train._mcp.clique_size':cs1,
            'data.train._mcp.edge_density':p1
            })
        mean_error = np.mean(l_errors)
        line = get_line(p1,p2,mean_error)
        add_line(filename,line)









