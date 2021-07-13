from inspect import GEN_CLOSED
from SBM_rfixed import MODEL_NAME
import loaders.data_generator as dg
import torch
import sklearn.metrics as skmetrics
import numpy as np
from toolbox.data_handler import DataHandler
import yaml
import os
import toolbox.utils as utils
from toolbox.data_handler import Planner
import tqdm
from models import get_model as get_model_gnn
from toolbox.helper import get_helper
from toolbox.optimizer import get_optimizer
from trainer import cycle_simple
import toolbox.searches as searches
import toolbox.metrics as metrics

CONFIG_FILE = 'article_config.yaml'
MODEL_PREFIX  = 'model-{}-' #
BASE_PATH   = './exps/'
MODEL_DIR_NAME = 'models/'
DATA_FILENAME = 'data-{}-n_{}.csv' #Experiment-n_vertices
utils.check_dir(BASE_PATH)
COLUMNS = []


# Experiment specific functions : these have switch cases depending on the problem at hand

def get_dataset(problem, train_test_val, config, value, num_workers=4):
    gen_args = dict()
    gen_args['n_vertices'] = config['n_vertices']
    gen_args['batch_size'] = config['model_specific']['GNN']['train']['batch_size']
    gen_args['num_examples_train'] = config['model_specific']['GNN']['train']['num_examples_train']
    gen_args['num_examples_val'] = config['model_specific']['GNN']['train']['num_examples_val']
    gen_args['num_examples_test'] = config['model_specific']['GNN']['train']['num_examples_test']
    for key in config['gen_specific']:
        gen_args[key] = config['gen_specific'][key]
    
    if problem=='hhc':
        gen = dg.HHC_Generator
        gen_args['cycle_param'] = 0
        gen_args['fill_param'] = value
    elif problem=='sbm':
        gen = dg.SBM_Generator
        c = gen_args['c']
        gen_args['p_inter'] = c-value/2
        gen_args['p_outer'] = c+value/2
    elif problem=='mcp':
        gen = dg.MCP_Generator
        gen_args['clique_size'] = value
    else:
        raise NotImplementedError(f'Problem {problem} not implemented')
    
    generator = gen(train_test_val, gen_args)
    generator.load_dataset()
    loader = torch.utils.data.DataLoader(generator, batch_size=gen_args['batch_size'], shuffle=True,num_workers=num_workers)
    return loader

def prepare_model(cur_value, config,device):
    experiment_name = config['experiment']
    problem = config['problem']
    model_type = config['model']
    train_value = config['train_value']
    n_vertices = config['n_vertices']

    model_prefix = MODEL_PREFIX.format(experiment_name)

    if model_type == 'GNN':
        if train_value=='retrain' or train_value=='r':
            train_value = cur_value
        model_name = model_prefix + 'GNN-n_{}-{}.tar'.format(n_vertices, train_value)
        model_path = os.path.join(BASE_PATH, experiment_name, MODEL_DIR_NAME, model_name)
        model_exists = check_model_exists(model_path)
        arch = config['model_specific']['GNN']['arch']
        model = get_model_gnn(arch)
        if model_exists:
            state_dict = load_model_dict(model_path,device)
            model.load_state_dict(state_dict)
            model.to(device)
        else:
            helper_object = get_helper(experiment_name)
            helper_config = config['model_specific']['GNN']
            helper = helper_object(config['name'],helper_config)
            optimizer, scheduler = get_optimizer(config['model_specific']['GNN']['train']['opt_args'],model)
            train_loader = get_dataset(problem,'train',config,cur_value)
            val_loader  = get_dataset(problem,'val',config,cur_value)
            cycle_simple(config['model_specific']['GNN']['train']['epoch'],train_loader,val_loader,model,optimizer,scheduler,helper,device)
            save_model(model_path,model)
    else:
        raise NotImplementedError(f'Model type {model_type} not implemented')
    
    return model

def get_exact_solver_function(problem):
    if problem == 'hhc':
        return searches.tsp_concorde
    elif problem == 'mcp':
        return searches.mc_bronk2_cpp
    elif problem == 'sbm':
        return searches.minb_kl_multiple
    else:
        raise NotImplementedError(f'Problem {problem} not implemented')

def get_value_function(problem):
    if problem == 'hhc':
        return searches.tsp_sym_value
    elif problem == 'mcp':
        return lambda _,adj: searches.mcp_clique_size(adj)
    elif problem == 'sbm':
        return searches.cut_value
    else:
        raise NotImplementedError(f'Problem {problem} not implemented')

def get_beam_search_function(problem):
    """Should take as arguments : 
        - data : the connectivity matrix
        - raw_scores : the raw output of the model"""
    if problem == 'hhc':
        return lambda data, raw_scores : searches.tsp_beam_decode(raw_scores, data)
    elif problem == 'mcp':
        return searches.mcp_beam_method
    elif problem == 'sbm':
        return lambda data, raw_scores : searches.sbm_get_adj(raw_scores)
    else:
        raise NotImplementedError(f'Problem {problem} not implemented')

def get_auc_function(problem):
    if problem == 'mcp':
        return metrics.mcp_auc
    elif problem in {'hhc','sbm'}:
        return base_auc
    else:
        raise NotImplementedError(f'Problem {problem} not implemented')

def get_overlap_function(problem):
    if problem == 'hhc':
        return overlap_p_np
    elif problem == 'mcp':
        return metrics.mcp_acc
    elif problem == 'sbm':
        return overlap_p_np
    else:
        raise NotImplementedError(f'Problem {problem} not implemented')


#####################

def get_config(file):
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    problem_name = data['experiment']
    clean_data = data['_'+problem_name]
    for key in data.keys():
        if not utils.is_pbm_key(key):
            clean_data[key] = data[key]
    return clean_data
        
def check_model_exists(model_filename)->bool:
    return os.path.isfile(model_filename)

def save_model(model_filename,model)->None:
    utils.check_file(model_filename)
    torch.save(model.state_dict(), model_filename)

def load_model_dict(model_filename,device):
    utils.check_file(model_filename)
    state_dict = torch.load(model_filename,map_location=device)
    return state_dict
    

def overlap_p_np(P,NP):
    bs,n,_ = P.shape
    true_pos = ((1 - (P-NP)**2)).sum()
    acc = true_pos/(bs*n*n)
    if isinstance(acc, torch.Tensor):
        acc = acc.item()
    return  acc # Should be a scalar

def base_auc(probas, reference):
    bs,n,_ = probas.shape
    return [skmetrics.roc_auc_score(reference[i].cpu().detach().reshape(n*n).numpy(), probas[i].cpu().detach().reshape(n*n).numpy()) for i in range(bs)]

def eval(model, dataset, exact_solver_function, overlap_function, beam_search_function, auc_function, value_function, device='cpu'):
    model.eval()
    model.to(device)

    o_pnp,o_gp,o_gnp = [],[],[]
    auc_gp,auc_gnp = [],[]
    v_p,v_np,v_g = [],[],[]

    total_tests = 0
    for data, P in dataset:
        bs,n,_ = P.shape

        adj = data[:,:,:,1] #Keep the connectivity/adjacency matrix before changing the device

        data = data.to(device)
        P = P.to(device)

        output = model(data)
        raw_scores = output.squeeze(-1)
        probas = torch.sigmoid(raw_scores)

        model_solution = beam_search_function(adj,raw_scores)
        if isinstance(model_solution[0],torch.Tensor):
            model_solution = utils.list_to_tensor(model_solution)

        NP = exact_solver_function(adj)

        opnp = overlap_function(P,NP)
        assert isinstance(opnp,float)
        o_pnp.append(opnp)
        ognp = overlap_function(model_solution,NP)
        assert isinstance(ognp,float)
        o_gnp.append(ognp)
        ogp = overlap_function(model_solution,P)
        assert isinstance(ogp,float)
        o_gp.append(ogp)

        #auc = [skmetrics.roc_auc_score(NP[i].cpu().detach().reshape(n*n).numpy(), probas[i].cpu().detach().reshape(n*n).numpy()) for i in range(bs)]
        auc = auc_function(probas,NP)
        auc_gnp.extend(auc)

        #auc = [skmetrics.roc_auc_score(P[i].cpu().detach().reshape(n*n).numpy(), probas[i].cpu().detach().reshape(n*n).numpy()) for i in range(bs)]
        auc = auc_function(probas,P)
        auc_gp.extend(auc)

        vnp = value_function(adj,NP)
        assert isinstance(vnp,float) or isinstance(vnp,int)
        v_np.append(vnp)
        vp = value_function(adj,P)
        assert isinstance(vp,float) or isinstance(vp,int)
        v_p.append(vp)
        vg = value_function(adj,model_solution)
        assert isinstance(vg,float) or isinstance(vg,int)
        v_g.append(vg)

    o_pnp = np.mean(o_pnp)
    o_gp  = np.mean(o_gp)
    o_gnp = np.mean(o_gnp)

    auc_gp = np.mean(auc_gp)
    auc_gnp= np.mean(auc_gnp)

    v_p = np.mean(v_p)
    v_np= np.mean(v_np)
    v_g = np.mean(v_g)
    value_dict = {'ol_gp':o_gp,'ol_gnp':o_gnp,'ol_pnp':o_pnp,'auc_gp':auc_gp,'auc_gnp':auc_gnp,'v_p':v_p,'v_np':v_np,'v_g':v_g}
    return value_dict


def handle_experiment(config):
    device = config['device']
    problem = config['problem']
    exp_name = config['experiment']
    n_vertices = config['n_vertices']
    path = os.path.join(BASE_PATH,exp_name)

    data_path = os.path.join(path, DATA_FILENAME.format(exp_name,n_vertices))

    planner = Planner(data_path)

    range_start = config['range_start']
    range_end  = config['range_end' ]
    range_steps = config['steps']

    key_variable = config['key_variable']

    values = np.linspace(range_start, range_end, range_steps)
    for value in values:
        planner.add_task((key_variable,value))
    
    exact_solver_function = get_exact_solver_function(problem)
    overlap_function = get_overlap_function(problem)
    beam_search_function = get_beam_search_function(problem)
    auc_function = get_auc_function(problem)
    value_function = get_value_function(problem)

    if planner.n_tasks==0:
        print("No tasks to be done, ending.")
        return

    progress_bar = tqdm.trange(planner.n_tasks)
    for _ in progress_bar:
        value_name,cur_value = planner.next_task()
        model = prepare_model(cur_value, config, device)
        test_loader = get_dataset(problem,'test',config,cur_value)
        value_dict = eval(model,test_loader, exact_solver_function, overlap_function, beam_search_function, auc_function, value_function, device=device)
        planner.add_entry_with_value(value_name,cur_value,value_dict)
    planner.write()



if __name__=='__main__':
    config = get_config(CONFIG_FILE)

    use_cuda = not config['cpu'] and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    config['device'] = device

    print('Using device:', device)
    handle_experiment(config)


