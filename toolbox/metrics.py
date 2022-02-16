import numpy as np
from numpy.lib.arraysetops import isin
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.modules.activation import Sigmoid, Softmax
from toolbox.utils import get_device, greedy_qap, perm_matrix
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sklearn.metrics as skmetrics
import toolbox.utils as utils

#from toolbox.searches import mcp_beam_method

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

def accuracy_linear_assignment(rawscores, dummy_target, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    total_n_vertices = 0
    acc = 0
    all_acc = []
    weights = torch.log_softmax(rawscores,-1)
    for i, weight in enumerate(weights):
        if labels:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        cost = -weight.cpu().detach().numpy()
        _, preds = linear_sum_assignment(cost)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc

def accuracy_max(weights,dummy_target, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    acc = 0
    all_acc = []
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        weight = weight.cpu().detach().numpy()
        preds = np.argmax(weight, 1)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc


def all_losses_acc(val_loader,model,criterion,
            device,eval_score=None):
    #model.eval()
    all_losses =[]
    all_acc = []

    for (data, target) in val_loader:
        data = data.to(device)
        target_deviced = target.to(device)
        output = model(data)
        rawscores = output
            
        loss = criterion(rawscores,target_deviced)
        
        all_losses.append(loss.item())
    
        if eval_score is not None:
            acc = eval_score(rawscores,target_deviced,aggregate_score=False)
            all_acc += acc
    return np.array(all_losses), np.array(all_acc)

# code below should be corrected/refactored...

def all_greedy_losses_acc(val_loader,model,criterion,
            device,T=10):
    # only tested with batch size = 1
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
        
        all_losses.append(loss.item())

        A = data[0,0,:,:,1].data.cpu().detach().numpy()
        B = data[0,1,:,:,1].data.cpu().detach().numpy()
        cost = -raw_scores.cpu().detach().numpy().squeeze()
        #print(i, " | ", cost)
        row, preds = linear_sum_assignment(cost)
        a, na,nb, acc, _ = greedy_qap(A,B,perm_matrix(row,preds),T)
        all_acc.append(acc)
    
    return np.array(all_losses), np.array(all_acc)
