import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

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

def make_meter_matching():
    meters_dict = {
        'loss': Meter(),
        'acc': Meter(),
        #'acc_gr': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def make_meter_tsp():
    meters_dict = {
        'loss': Meter(),
        'f1': Meter(),
        #'acc_gr': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def accuracy_linear_assignment(weights,labels=None,aggregate_score = True):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    total_n_vertices = 0
    acc = 0
    all_acc = []
    for i, weight in enumerate(weights):
        if labels:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        cost = -weight.cpu().detach().numpy()
        _ , preds = linear_sum_assignment(cost)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
            return acc, total_n_vertices
        else:
            all_acc.append(np.sum(preds == label)/len(weight))
            return all_acc

    

def accuracy_max(weights,labels=None):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    acc = 0
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        if labels:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        weight = weight.cpu().detach().numpy()
        preds = np.argmax(weight, 1)
        #print(preds)
        acc += np.sum(preds == label)
        total_n_vertices += len(weight)
    return acc, total_n_vertices


def f1_score(preds,labels,device = 'cuda'):
    """
    take 2 adjacency matrices and compute precision, recall, f1_score for a tour
    """
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

def compute_f1(raw_scores,target,device):
    _, ind = torch.topk(raw_scores, 3, dim =2)
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)
    return f1_score(y_onehot,target)