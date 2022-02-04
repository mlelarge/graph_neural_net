import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
from toolbox.utils import get_device


class triplet_loss(nn.Module):
    def __init__(self, loss_reduction='mean', loss=nn.CrossEntropyLoss(reduction='sum')):
        super(triplet_loss, self).__init__()
        self.loss = loss
        if loss_reduction == 'mean':
            self.increments = lambda new_loss, n_vertices : (new_loss, n_vertices)
        elif loss_reduction == 'mean_of_mean':
            self.increments = lambda new_loss, n_vertices : (new_loss/n_vertices, 1)
        else:
            raise ValueError('Unknown loss_reduction parameters {}'.format(loss_reduction))

    def forward(self, raw_scores, target_dummy):#Keep target for modularity, target should be an empty tensor, but it's not used anyways
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        device = get_device(raw_scores)
        loss = 0
        total = 0
        for out in raw_scores:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(device)
            incrs = self.increments(self.loss(out, target), n_vertices)
            loss += incrs[0]
            total += incrs[1]
        return loss/total
