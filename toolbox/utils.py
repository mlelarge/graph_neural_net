import os
import json
import torch
import numpy as np

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_device(t):
    if t.is_cuda:
        return t.get_device()
    return 'cpu'

def permute_adjacency_twin(t1,t2):
    """
    Makes a permutation of an adjacency matrix. Equivalent to a renaming of the nodes.
    Supposes shape (n,n)
    """
    n,_ = t1.shape
    perm = torch.randperm(n)
    return t1[perm,:][:,perm],t2[perm,:][:,perm]

def mcp_adj_to_ind(adj):
    """
    adj should be of size (n,n) or (bs,n,n), supposedly the solution for mcp
    Transforms the adjacency matrices in a list of indices corresponding to the clique
    """
    solo=False
    if len(adj.shape)==2:
        solo=True
        adj = adj.unsqueeze(0)
    bs,n,_ = adj.shape
    sol_onehot = torch.sum(adj,dim=-1)#Gets the onehot encoding of the solution clique
    l_sol_indices = [torch.where(sol_onehot[i])[0] for i in range(bs)] #Converts the onehot encoding to a list of the nodes' numbers
    l_clique_sol = [{elt.item() for elt in indices} for indices in l_sol_indices]
    
    if solo:
        l_clique_sol=l_clique_sol[0]
    return l_clique_sol

def mcp_ind_to_adj(ind,n):
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
    adj *= 0*torch.eye(n)
    return adj