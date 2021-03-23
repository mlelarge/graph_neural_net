import torch
import torch.nn as nn
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

    def forward(self, outputs, target):#Keep target for modularity, target should be an empty tensor, but it's not used anyways
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        device = get_device(outputs)
        loss = 0
        total = 0
        for out in outputs:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(device)
            incrs = self.increments(self.loss(out, target), n_vertices)
            loss += incrs[0]
            total += incrs[1]
        return loss/total

# TODO refactor

class tsp_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='none'), normalize=nn.Sigmoid()):
        super(tsp_loss, self).__init__()
        self.loss = loss
        self.normalize = normalize
        
    def forward(self, raw_scores, target):
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        proba = self.normalize(raw_scores)
        loss = self.loss(proba,target)
        return torch.mean(loss)

class tsp_rl_loss(nn.Module):
    
    def __init__(self, normalize=nn.Softmax(dim=-1)):
        super(tsp_rl_loss, self).__init__()
        self.normalize = normalize
    
    def forward(self, raw_scores, distance_matrix):
        """
        raw_scores (bs,n_vertices,n_vertices), target (bs,n,n) and should be the distance matrix W
        """
        proba = self.normalize(raw_scores)
        #print(proba)
        proba = proba*proba.transpose(-2,-1) #Symmetrization
        loss = torch.sum(torch.sum(proba*distance_matrix, dim=-1), dim=-1)
        return torch.mean(loss)


class mcp_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='mean'), normalize=nn.Sigmoid()):
        super(mcp_loss, self).__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, raw_scores, target):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        preds = self.normalize(raw_scores)
        loss = self.loss(preds,target)
        return torch.mean(loss)


class sbm_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='none'), normalize=nn.Sigmoid()):
        super(sbm_loss, self).__init__()
        self.loss = loss
        self.normalize = normalize
        
    def forward(self, raw_scores, target):
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        proba = self.normalize(raw_scores)
        loss = self.loss(proba,target)
        return torch.mean(loss)