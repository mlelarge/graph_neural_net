import os
import shutil
import json
from typing import Tuple
from matplotlib.pyplot import isinteractive
from numpy.lib.arraysetops import isin
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from networkx import to_numpy_array as nx_to_numpy_array
import dgl as dgl
import torch.backends.cudnn as cudnn

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

def check_file(file_path):
    file_path = file_path.replace('//','/')
    dir_path = os.path.dirname(file_path)
    check_dir(dir_path)
    if not os.path.exists(file_path):
        with open(file_path,'w') as f:
            pass

def setup_env(cpu):
    # Randomness is already controlled by Sacred
    # See https://sacred.readthedocs.io/en/stable/randomness.html
    if not cpu:
        cudnn.benchmark = True

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    #check_dir(log_dir)
    filename = os.path.join(log_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(log_dir, 'model_best.pth.tar'))
        #shutil.copyfile(filename, model_path)
        print(f"Best Model yet : saving at {log_dir+'model_best.pth.tar'}")

    fn = os.path.join(log_dir, 'checkpoint_epoch{}.pth.tar')
    torch.save(state, fn.format(state['epoch']))

    if (state['epoch'] - 1 ) % 5 != 0:
      #remove intermediate saved models, e.g. non-modulo 5 ones
      if os.path.exists(fn.format(state['epoch'] - 1 )):
          os.remove(fn.format(state['epoch'] - 1 ))

    state['exp_logger'].to_json(log_dir=log_dir,filename='logger.json')

# move in utils
def load_model(model, device, model_path):
    """ Load model. Note that the model_path argument is captured """
    if os.path.exists(model_path):
        print("Reading model from ", model_path)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        raise RuntimeError('Model does not exist!')

def save_to_json(jsonkey, loss, relevant_metric_dict, filename):
    if os.path.exists(filename):
        with open(filename, "r") as jsonFile:
            data = json.load(jsonFile)
    else:
        data = {}
    data[jsonkey] = {'loss':loss}
    for dkey, value in relevant_metric_dict.items():
        data[jsonkey][dkey] = value
    with open(filename, 'w') as jsonFile:
        json.dump(data, jsonFile)

# from https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

#def is_pbm_key(chars):
#    """
#    Recognizes problem keys of the form '_*' (Ex: '_mcp','_tsp')
#    """
#    return isinstance(chars,str) and chars[0]=='_'

# def clean_config(config,pbm_key):
#     if not isinstance(config,dict):
#         return config
#     new_config = {}
#     for k,v in config.items():
#         if not(is_pbm_key(k)) or k==pbm_key:
#             new_config[k] = clean_config(v,pbm_key)
#     return new_config

# def clean_config_inplace(config,pbm_key):
#     assert isinstance(config,dict), "Trying to clean something that is not a dictionary!"
#     keys_to_delete = []
#     for k,v in config.items():
#         if is_pbm_key(k) and k!=pbm_key:
#             keys_to_delete.append(k)
#         elif isinstance(v,dict):
#             clean_config_inplace(v,pbm_key)
#     for k in keys_to_delete:
#         config.pop(k)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_device(t):
    if t.is_cuda:
        return t.get_device()
    return 'cpu'

#Matrix operation

def symmetrize_matrix(A):
    """
    Symmetrizes a matrix :
    If shape is (a,b,c) will symmetrize by considering a is batch size
    """
    Af = A.triu(0) + A.triu(1).transpose(-2,-1)
    return Af

def list_to_tensor(liste) -> torch.Tensor:
    """Transforms a list of same shaped tensors"""
    if isinstance(liste,torch.Tensor):
        return liste
    bs = len(liste)
    shape = liste[0].shape
    final_shape = (bs,*shape)
    tensor_eq = torch.empty(final_shape)
    for k in range(bs):
        tensor_eq[k] = liste[k]
    return tensor_eq

#Graph operations

def edge_features_to_dense_tensor(graph, features, device='cpu'):
    N = graph.number_of_nodes()
    resqueeze = False
    if len(features.shape)==1:
        features.unsqueeze(-1)
        resqueeze = True
    n_feats = features.shape[1]
    t = torch.zeros((N,N,n_feats)).to(device)
    #adj = torch.tensor(nx_to_numpy_array(graph.to_networkx())).to(device)#edges = np.array(graph.edges().cpu()).T #Transpose for the right shape (2,n_edges)
    adj = graph.adj(ctx=device).to_dense()
    ix,iy = torch.where(adj==1)
    t[ix,iy] = features
    if resqueeze:
        t.squeeze(-1)
    return t

def edge_features_to_dense_sym_tensor(graph,features,device='cpu'):
    t = edge_features_to_dense_tensor(graph,features,device)
    if torch.all(t.transpose(0,1)+t==2*t): #Matrix already symmetric
        return t
    
    N = graph.number_of_nodes()
    tril = torch.tril(torch.ones((N,N)),-1)
    tril = tril.unsqueeze(-1).to(device) #For the multiplication, we need to add the dimension
    if torch.all(t*tril==0): #Only zeros in the lower triangle features
        return t + t.transpose(0,1) * tril #Here we remove the diagonal with '* tril'
    
    tbool = (t!=0)
    tbool = tbool.sum(-1)!=0 #Here we have True where the feature vectors are not 0
    ix,iy = torch.where(tbool!=0)
    for i,j in zip(ix,iy):
        if i==j or torch.all(t[j,i]==t[i,j]):
            continue
        elif torch.all(t[j,i]==0):
            t[j,i] = t[i,j]
        else:
            raise AssertionError(f"Feature values are asymmetric, should not have used the symetric function.")
    return t

def edge_features_to_dense_features(graph, features, device='cpu'):
    t = edge_features_to_dense_tensor(graph, features, device)
    if len(features.shape)==1:
        return t.flatten()
    n_features = features.shape[1]
    N = graph.number_of_nodes()
    t_features = t.reshape((N**2,n_features))
    return t_features

def edge_features_to_dense_sym_features(graph, features, device='cpu'):
    t = edge_features_to_dense_sym_tensor(graph, features, device)
    if len(features.shape)==1:
        return t.flatten()
    n_features = features.shape[1]
    N = graph.number_of_nodes()
    t_features = t.reshape((N**2,n_features))
    return t_features

def edge_tensor_to_features(graph: dgl.DGLGraph, features: torch.Tensor, device='cpu'):
    n_edges = graph.number_of_edges()
    resqueeze = False
    if len(features.shape)==3:
        resqueeze=True
        features = features.unsqueeze(-1)
    bs,N,_,n_features = features.shape
    
    ix,iy = graph.edges()
    bsx,bsy = ix//N,iy//N
    Nx,Ny = ix%N,iy%N
    assert torch.all(bsx==bsy), "Edges between graphs, should not be allowed !" #Sanity check
    final_features = features[(bsx,Nx,Ny)] #Here, shape will be (n_edges,n_features)
    if resqueeze:
        final_features = final_features.squeeze(-1)
    return final_features

def temp_sym(t):
    if torch.all(t.transpose(0,1)+t==2*t):
        return t
    elif torch.all(torch.tril(t,-1)==0):
        return t + torch.triu(t,1).transpose(0,1)
    else:
        ix,iy = torch.where(t!=0)
        for i,j in zip(ix,iy):
            if t[j,i]==0:
                t[j,i] = t[i,j]
            elif t[j,i]==t[i,j]:
                continue
            else:
                raise AssertionError(f"Feature values are asymmetric, should not have used the symetric function.")
    return t

#QAP

def perm_matrix(row,preds):
    n = len(row)
    permutation_matrix = np.zeros((n, n))
    permutation_matrix[row, preds] = 1
    return permutation_matrix

def score(A,B,perm):
    return np.trace(A @ perm @ B @ np.transpose(perm))/2, np.sum(A)/2, np.sum(B)/2

def improve(A,B,perm):
    label = np.arange(A.shape[0])
    cost_adj = - A @ perm @ B
    r, p = linear_sum_assignment(cost_adj)
    acc = np.sum(p == label)
    return perm_matrix(r,p), acc

def greedy_qap(A,B,perm,T,verbose=False):
    #perm_p = perm
    s_best, na, nb = score(A,B,perm) 
    perm_p, acc_best = improve(A,B,perm)
    T_best = 0
    for i in range(T):
        perm_n, acc = improve(A,B,perm_p)
        perm_p = perm_n
        s,na,nb = score(A,B,perm_p)
        if s > s_best:
            acc_best = acc
            s_best = s
            T_best = i
        if verbose:
            print(s,na,nb,acc)
    return s_best, na, nb, acc_best, T_best

#MCP

def get_cs(t):
    """ 
    Returns the clique size of target tensor t
    """
    return torch.sum(torch.sign(torch.sum(t,dim=-1))).to(int).item()

def permute_adjacency_twin(t1,t2) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Makes a permutation of two adjacency matrices together. Equivalent to a renaming of the nodes.
    Supposes shape (n,n)
    """
    n,_ = t1.shape
    perm = torch.randperm(n)
    return t1[perm,:][:,perm],t2[perm,:][:,perm]

def permute_adjacency(t,perm=None) -> torch.Tensor:
    """
    Makes a permutation of an adjacency matrix. Equivalent to a renaming of the nodes.
    Supposes shape (n,n)
    If perm is not specified, it randomly permutes
    """
    n,_ = t.shape
    if perm is None:
        perm = torch.randperm(n)
    else:
        perm = perm.to(int)
    return t[perm,:][:,perm]

def mcp_adj_to_ind(adj)->list:
    """
    adj should be of size (n,n) or (bs,n,n), supposedly the solution for mcp
    Transforms the adjacency matrices in a list of indices corresponding to the clique
    """
    solo=False
    if len(adj.shape)==2:
        solo=True
        adj = adj.unsqueeze(0)
    bs,_,_ = adj.shape
    sol_onehot = torch.sum(adj,dim=-1)#Gets the onehot encoding of the solution clique
    l_sol_indices = [torch.where(sol_onehot[i]!=0)[0] for i in range(bs)] #Converts the onehot encoding to a list of the nodes' numbers
    l_clique_sol = [{elt.item() for elt in indices} for indices in l_sol_indices]
    
    if solo:
        l_clique_sol=l_clique_sol[0]
    return l_clique_sol

def mcp_ind_to_adj(ind,n)->torch.Tensor:
    """
    ind should be a set of indices (or iterable)
    Transforms it into the adjacency matrix of shape (n,n)
    """
    assert max(ind)<n, f"Index {max(ind)} not in range for {n} indices"
    adj = torch.zeros((n,n))
    n_indices = len(ind)
    x = [elt for elt in ind for _ in range(n_indices)]
    y = [elt for _ in range(n_indices) for elt in ind]
    adj[x,y] = 1
    adj *= (1-torch.eye(n))
    return adj


#TSP

# Why that?
def reduce_name(pbmkey):
    if pbmkey[:3]=='tsp':
        pbmkey='tsp'
    return pbmkey

def tsp_get_min_dist(dist_matrix):
    """
    Takes a distance matrix dist_matrix of shape (n,n) or (bs,n,n)
    Returns the mean minimal distance between two distinct nodes for each of the *bs* instances 
    """
    if not isinstance(dist_matrix,torch.Tensor):
        try:
            dist_matrix = torch.tensor(dist_matrix)
        except Exception:
            raise ValueError(f"Type {type(dist_matrix)} could not be broadcasted to torch.Tensor")
    solo=False
    if len(dist_matrix.shape)==2:
        solo=True
        dist_matrix = dist_matrix.unsqueeze(0)
    bs,n,_ = dist_matrix.shape
    max_dist = torch.max(dist_matrix)
    eye_mask = torch.zeros_like(dist_matrix)
    eye_mask[:] = torch.eye(n)
    dist_modif = dist_matrix + max_dist*eye_mask
    min_dist = torch.min(dist_modif,dim=-1).values
    min_mean = torch.mean(min_dist,dim=-1)
    if solo:
        min_mean = min_mean[0]
    
    return min_mean

def points_to_dist(xs,ys)->np.array:
    n=len(xs)
    points = np.zeros((n,2))
    points[:,0] = xs
    points[:,1] = ys
    dist_matrix = cdist(points,points)
    return dist_matrix

def is_permutation_matrix(x):
    '''
    Checks if x is a permutation matrix, thus a tour.
    '''
    x = x.squeeze()
    return (x.ndim == 2 and x.shape[0] == x.shape[1] and
            (x.sum(dim=0) == 1).all() and 
            (x.sum(dim=1) == 1).all() and
            ((x == 1) | (x == 0)).all())

def tour_to_perm(n,path):
    '''
    Transform a tour into a permutation (directed tour)
    '''
    m = torch.zeros((n,n))
    for k in range(n):
        u = int(path[k])
        v = int(path[(k+1)%n])
        m[u,v] = 1
    return m

def tour_to_adj(n,path):
    '''
    Transform a tour into an adjacency matrix
    '''
    m = torch.zeros((n,n))
    for k in range(n):
        u = int(path[k])
        v = int(path[(k+1)%n])
        m[u,v] = 1
        m[v,u] = 1
    return m

#SBM

def part_to_adj(p1,p2):
    n = len(p1)+len(p2)
    adj1 = mcp_ind_to_adj(p1,n)
    adj2 = mcp_ind_to_adj(p2,n)
    return adj1 + adj2 + torch.eye(n)


