import numpy as np

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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_sum(self):
        return self.sum

def make_meter_matching():
    meters_dict = {
        'loss': Meter(),
        'acc_la': Meter(),
        'acc_max': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def accuracy_linear_assigment(weights,labels):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    bs = weights.shape[0]
    n = weights.shape[1]
    acc = 0
    for i in range(bs):
        cost = -weights[i,:,:]
        _ , preds = linear_sum_assignment(cost)
        acc += np.sum(preds == labels[i,:])
    return acc/(n*bs)

def accuracy_max(weights,labels):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    bs = weightd.shape[0]
    n = weights.shape[1]
    acc = 0
    for i in range(bs):
        preds = np.armax(weights[i,:,:], 2)
        acc += np.sum(preds == labels[i,:])
    return acc/(n*bs)