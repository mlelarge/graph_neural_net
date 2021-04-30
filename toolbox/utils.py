import os
import json
from typing import Tuple
from matplotlib.pyplot import isinteractive
from numpy.lib.arraysetops import isin
import torch
import numpy as np
from scipy.spatial.distance import cdist

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

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

def is_pbm_key(chars):
    """
    Recognizes problem keys of the form '_*' (Ex: '_mcp','_tsp')
    """
    return isinstance(chars,str) and chars[0]=='_'

def clean_config(config,pbm_key):
    if not isinstance(config,dict):
        return config
    new_config = {}
    for k,v in config.items():
        if not(is_pbm_key(k)) or k==pbm_key:
            new_config[k] = clean_config(v,pbm_key)
    return new_config

def clean_config_inplace(config,pbm_key):
    assert isinstance(config,dict), "Trying to clean something that is not a dictionary!"
    keys_to_delete = []
    for k,v in config.items():
        if is_pbm_key(k) and k!=pbm_key:
            keys_to_delete.append(k)
        elif isinstance(v,dict):
            clean_config_inplace(v,pbm_key)
    for k in keys_to_delete:
        config.pop(k)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_device(t):
    if t.is_cuda:
        return t.get_device()
    return 'cpu'


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




