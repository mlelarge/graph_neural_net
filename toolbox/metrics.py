import numpy as np
from numpy.lib.arraysetops import isin
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.modules.activation import Sigmoid, Softmax
from toolbox.utils import get_device
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sklearn.metrics as skmetrics
import toolbox.utils as utils

from toolbox.searches import mcp_beam_method

class Meter(object):
    """Computes and stores the sum, average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_sum(self):
        return self.sum
    
    def value(self):
        """ Returns the value over one epoch """
        return self.avg

    def is_active(self):
        return self.count > 0

class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val

def make_meter_loss():
    meters_dict = {
        'loss': Meter(),
        'loss_ref': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def make_meter_acc():
    meters_dict = {
        'loss': Meter(),
        'acc': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def make_meter_f1():
    meters_dict = {
        'loss': Meter(),
        'f1': Meter(),
        'precision': Meter(),
        'recall': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

#QAP

def accuracy_linear_assignment(weights, dummy_target, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    total_n_vertices = 0
    acc = 0
    all_acc = []
    #print(target)
    for i, weight in enumerate(weights):
        if labels:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        cost = -weight.cpu().detach().numpy()
        #print(i, " | ", cost)
        _, preds = linear_sum_assignment(cost)
        if aggregate_score:
            acc += np.sum(preds == label)
            #print(i, " | ", acc, preds, label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc

def all_losses_acc(val_loader,model,criterion,
            device,eval_score=None):
    model.eval()
    all_losses =[]
    all_acc = []

    for (data, target) in val_loader:
        data = data.to(device)
        target_deviced = target.to(device)
        output = model(data)
        rawscores = output.squeeze(-1)
        raw_scores = torch.softmax(rawscores,-1)
            
        loss = criterion(raw_scores,target_deviced)
        #input1 = input1.to(device)
        #input2 = input2.to(device)
        #output = model(input1,input2)

        #loss = criterion(output)
        #print(output.shape)
        all_losses.append(loss.item())
    
        if eval_score is not None:
            acc = eval_score(raw_scores,target_deviced,aggregate_score=False)#eval_score(output, aggregate_score=False)
            all_acc += acc
    return np.array(all_losses), np.array(all_acc)
   
def accuracy_max(weights,dummy_target, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    acc = 0
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        weight = weight.cpu().detach().numpy()
        preds = np.argmax(weight, 1)
        #print(preds)
        acc += np.sum(preds == label)
        total_n_vertices += len(weight)
    if aggregate_score:
        return acc, total_n_vertices
    else:
        return acc/total_n_vertices
    #return acc, total_n_vertices

#MCP

def accuracy_max_mcp(weights,clique_size):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    true_pos = 0
    false_pos = 0
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        weight = weight.cpu().detach().numpy()
        #print(weight)
        deg = np.sum(weight, 0)
        inds = np.argpartition(deg, -clique_size)[-clique_size:]
        #print(preds)#, np.sum(preds[:clique_size] <= clique_size))
        true_pos += np.sum(inds <= clique_size)
        #false_pos += np.sum(preds[clique_size:] < 0.1*clique_size)
        total_n_vertices += clique_size#len(weight)
    return true_pos, total_n_vertices

def accuracy_mcp(raw_scores,solutions):
    """
    weights and solutions should be (bs,n,n)
    """
    clique_sizes,_ = torch.max(solutions.sum(dim=-1),dim=-1) #The '+1' is because the diagonal of the solutions is 0
    clique_sizes += 1
    bs,n,_ = raw_scores.shape
    true_pos = 0
    total_n_vertices = 0

    probas = torch.sigmoid(raw_scores)

    deg = torch.sum(probas, dim=-1)
    inds = [ (torch.topk(deg[k],int(clique_sizes[k].item()),dim=-1))[1] for k in range(bs)]
    for i,_ in enumerate(raw_scores):
        sol = torch.sum(solutions[i],dim=1) #Sum over rows !
        ind = inds[i]
        for idx in ind:
            idx = idx.item()
            if sol[idx]:
                true_pos += 1
        total_n_vertices+=clique_sizes[i].item()
    return true_pos, total_n_vertices

def accuracy_mcp_exact(raw_scores,cliques_solutions):
    """
    weights should be (bs,n,n)
    cliques_solutions should be a list of size bs of lists of sets (multiple possible max cliques for each problem)
    """
    bs,n,_ = raw_scores.shape
    clique_sizes = torch.zeros(bs)
    for k,cliques_solution in enumerate(cliques_solutions):
        if len(cliques_solution)!=0:
            clique_sizes[k] = len(cliques_solution[0])

    probas = torch.sigmoid(raw_scores)
    deg = torch.sum(probas, dim=-1)
    inds = [ (torch.topk(deg[k],int(clique_sizes[k].item()),dim=-1))[1] for k in range(bs)]

    true_pos = 0
    total_n_vertices = 0
    l_sol_cliques = []
    for k,_ in enumerate(raw_scores):
        ind = inds[k]
        ind_set = set([elt.item() for elt in ind])
        possible_solutions = cliques_solutions[k]
        max_pos = 0
        best_clique = set()
        for sol in possible_solutions:
            cur_pos = len(sol.intersection(ind_set))
            if cur_pos>=max_pos:
                max_pos=cur_pos
                best_clique = sol
        true_pos+=max_pos
        total_n_vertices+=clique_sizes[k].item()
        l_sol_cliques.append(best_clique)
    return true_pos, total_n_vertices, l_sol_cliques

def mcp_auc(probas,cliques_solutions):
    bs,n,_ = probas.shape
    aucs = []
    for k in range(bs):
        best_auc_value = 0
        for _ in cliques_solutions[k]:
            clique = cliques_solutions[k]
            if isinstance(clique[0], set):
                clique = utils.mcp_ind_to_adj(clique[0],n)
            cur_auc = skmetrics.roc_auc_score(clique.reshape(n*n).numpy(), probas[k].cpu().detach().reshape(n*n).numpy())
            best_auc_value = max(best_auc_value, cur_auc)
        aucs.append(best_auc_value)
    return aucs

def accuracy_inf_sol(inferred,cliques_solution):
    """
    'inferred' should be a set of vertices
    'cliques_solution' an iterable of all the solution cliques (as sets)
    """
    assert len(cliques_solution)!=0, "No solution provided!"
    max_overlap = 0
    best_clique_sol = cliques_solution[0]
    clique_size = len(cliques_solution[0])
    for cur_clique in cliques_solution:
        temp_inter = cur_clique.intersection(inferred)
        cur_overlap = len(temp_inter)
        if cur_overlap > max_overlap:
            max_overlap = cur_overlap
            best_clique_sol = cur_clique
    return max_overlap, clique_size, best_clique_sol

def accuracy_inf_sol_multiple(inferred,cliques_solutions):
    """
    Batch sized version of accuracy_inf_sol
    """
    bs = len(inferred)
    true_pos = 0
    n_tot = 0
    l_sol_cliques = []
    for k in range(bs):
        cur_tp, cur_n, best_clique = accuracy_inf_sol(inferred[k],cliques_solutions[k])
        true_pos += cur_tp
        n_tot += cur_n
        l_sol_cliques.append(best_clique)
    return true_pos, n_tot, l_sol_cliques

def mcp_acc(inferred, cliques_or_target):
    def convert(t):
        if isinstance(t[0],torch.Tensor):
            if len(t[0].shape)>1:
                t = utils.mcp_adj_to_ind(t)
                t = [[elt] for elt in t] # Add a depth to have a shape of bs,1,set_size, as we work with possible multiple set solutions
        return t


    cliques_or_target = convert(cliques_or_target)
    #if (not isinstance(cliques_or_target,set)):
    #    if isinstance(cliques_or_target,list):
    #        cliques_or_target = utils.list_to_tensor(cliques_or_target)
    #    cliques_or_target = utils.mcp_adj_to_ind(cliques_or_target)
    tp,n_tot,_ = accuracy_inf_sol_multiple(inferred, cliques_or_target)
    return tp/n_tot

#TSP

def f1_score(preds,labels):
    """
    take 2 adjacency matrices and compute precision, recall, f1_score for a tour
    """
    device = get_device(preds)

    labels = labels.to(device)
    bs, n_nodes ,_  = labels.shape
    true_pos = 0
    false_pos = 0
    mask = torch.ones((n_nodes,n_nodes))-torch.eye(n_nodes)
    mask = mask.to(device)
    for i in range(bs):
        true_pos += torch.sum(mask*preds[i,:,:]*labels[i,:,:]).cpu().item()
        false_pos += torch.sum(mask*preds[i,:,:]*(1-labels[i,:,:])).cpu().item()
        #pos += np.sum(preds[i][0,:] == labels[i][0,:])
        #pos += np.sum(preds[i][1,:] == labels[i][1,:])
    #prec = pos/2*n
    prec = true_pos/(true_pos+false_pos)
    rec = true_pos/(2*n_nodes*bs)
    if prec+rec == 0:
        f1 = 0.0
    else:
        f1 = 2*prec*rec/(prec+rec)
    return prec, rec, f1#, n, bs

def compute_f1(raw_scores,target,k_best=3):
    """
    Computes F1-score with the k_best best edges per row
    For TSP with the chosen 3 best, the best result will be : prec=2/3, rec=1, f1=0.8 (only 2 edges are valid)
    """
    device = get_device(raw_scores)
    _, ind = torch.topk(raw_scores, k_best, dim = 2) #Here chooses the 3 best choices
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)
    return f1_score(y_onehot,target)

def tsp_rl_loss(raw_scores, distance_matrix):
    proba = Softmax(dim=-1)(raw_scores)
    proba = proba*proba.transpose(-2,-1)
    loss = torch.sum(torch.sum(proba*distance_matrix, dim=-1), dim=-1)
    return torch.mean(loss).data.item()

def tspd_dumb(raw_scores, target):
    """
    raw_scores and target of shape (bs,n,n)
    Just takes the order of the first column, it's a bit naive.
    As the data is ordered, it should be [0,1/n,2/n,...,1-1/n,1,1,1-1/n,...,1/n,0]
    The procedure is ordering the vertices and comparing directly
    """
    bs,n,_ = raw_scores.shape
    _,order = torch.topk(raw_scores,n,dim=2)
    results = order[:,0,:] #Keep the first row everytime
    true_pos=0
    for result in results:
        true_result = torch.cat( (result[::2],result[1::2].flip(-1)) )
        comparison = true_result==torch.arange(n)
        positives = torch.sum(comparison.to(int)).item()
        true_pos+=positives
    return true_pos,bs*n

#HHC
def accuracy_hhc(raw_scores, target):
    """ Computes simple accuracy by choosing the most probable edge
    For HHC:    - raw_scores and target of shape (bs,n,n)
                - target should be ones over the diagonal for the normal HHC (but can be changed to fit another solution)
     """
    bs,n,_ = raw_scores.shape
    device = get_device(raw_scores)
    _, ind = torch.topk(raw_scores, 1, dim = 2) #Here chooses the best choice
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)
    accu = target*y_onehot #Places 1 where the values are the same
    true_pos = torch.count_nonzero(accu).item()
    n_total = bs * n #Perfect would be that we have the right permutation for every bs 
    return true_pos,n_total

def perf_hhc(raw_scores, target):
    """ Computes the probability of getting the full HHC right
     """
    bs,n,_ = raw_scores.shape
    device = get_device(raw_scores)
    _, ind = torch.topk(raw_scores, 1, dim = 2) #Here chooses the best choice
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)
    accu = target*y_onehot #Places 1 where the values are the same
    true_pos = 0 #Will count the number of times we recovered the HHC
    for acc in accu:
        if torch.count_nonzero(acc).item()==n:
            true_pos+=1
    return true_pos,bs

    

#SBM
def accuracy_sbm_two_categories(raw_scores,target):
    """
    Computes a simple category accuracy
    Needs raw_scores.shape = (bs,n,out_features) and target.shape = (bs,n,n)
    """
    device = get_device(raw_scores)

    bs,n,_ = raw_scores.shape

    target_nodes = target[:,:,0] #Keep the category of the first node as 1, and the other at 0
    
    true_pos = 0

    embeddings = F.normalize(raw_scores,dim=-1) #Compute E
    similarity = embeddings @ embeddings.transpose(-2,-1) #Similarity = E@E.T
    for batch_embed,target_node in zip(similarity,target_nodes):
        kmeans = KMeans(n_clusters=2).fit(batch_embed.cpu().detach().numpy())
        labels = torch.tensor(kmeans.labels_).to(device)
        poss1 = torch.sum((labels==target_node).to(int))  
        poss2 = torch.sum(((1-labels)==target_node).to(int))
        best = max(poss1,poss2)
        #labels = 2*labels -1 #Normalize categories to 1 and -1
        #similarity = labels@labels.transpose(-2,-1)
        true_pos += int(best)
    return true_pos, bs * n

def accuracy_sbm_two_categories_edge(raw_scores,target):
    """
    Computes a simple category accuracy
    Needs raw_scores.shape = (bs,n,n) and target.shape = (bs,n,n)
    """
    device = get_device(raw_scores)

    bs,n,_ = raw_scores.shape

    #probas = torch.sigmoid(raw_scores) #No need for proba, we just need the best choices

    _,ind = torch.topk(raw_scores, n//2, -1)
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)

    true_pos = bs * n * n - int(torch.sum(torch.abs(target-y_onehot)))
    return true_pos, bs * n * n