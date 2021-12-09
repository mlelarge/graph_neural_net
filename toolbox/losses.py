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
        try:
            loss = self.loss(proba,target)
        except RuntimeError:
            loss = self.loss(proba,target.to(torch.long))
        return torch.mean(loss)

class tspd_loss(nn.Module):
    def __init__(self, loss=nn.MSELoss(reduction='none'), normalize=Sigmoid):
        super(tspd_loss, self).__init__()
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
        raw_scores =  raw_scores * raw_scores.transpose(-2,-1) #Symmetrization before proba
        #raw_scores *= 1-torch.eyer(raw_scores.shape[-1])
        proba = self.normalize(raw_scores)
        #print(proba)
        #proba = proba*proba.transpose(-2,-1) #Symmetrization
        loss = torch.sum(torch.sum(proba*distance_matrix, dim=-1), dim=-1) #In need of normalization of the loss
        return torch.mean(loss)

class hhc_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='none'), normalize=nn.Sigmoid()):
        super(hhc_loss, self).__init__()
        self.loss = loss
        self.normalize = normalize
    
    def forward(self, raw_scores, target):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        preds = self.normalize(raw_scores)
        loss = self.loss(preds,target)
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


class sbm_edge_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='none'), normalize = nn.Sigmoid()):
        super(sbm_edge_loss, self).__init__()
        self.loss = loss
        self.normalize = normalize    
    
    def forward(self, raw_scores, target):
        """
        raw_scores of shape (bs,n_vertices,out_features), target of shape (bs,n_vertices,n_vertices)
        """
        probas = self.normalize(raw_scores)
        loss = self.loss(probas,target)
        mean_loss = torch.mean(loss)
        return mean_loss

class sbm_node_loss(nn.Module):
    def __init__(self, loss=nn.MSELoss(reduction='none'), normalize = nn.Sigmoid()):
        super(sbm_node_loss, self).__init__()
        self.loss = loss
        self.normalize = normalize    
    
    def forward(self, raw_scores, target):
        """
        raw_scores of shape (bs,n_vertices,out_features), target of shape (bs,n_vertices,n_vertices)
        """
        embeddings = F.normalize(raw_scores,dim=-1) #Compute E
        similarity = embeddings @ embeddings.transpose(-2,-1) #Similarity = E@E.T
        loss = self.loss(similarity,target)
        mean_loss = torch.mean(loss)
        return mean_loss

