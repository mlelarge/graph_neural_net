import torch
import torch.nn as nn


class triplet_loss(nn.Module):
    def __init__(self, device='cpu', loss_reduction='mean', loss=nn.CrossEntropyLoss(reduction='sum')):
        super(triplet_loss, self).__init__()
        self.device = device
        self.loss = loss
        if loss_reduction == 'mean':
            self.increments = lambda new_loss, n_vertices : (new_loss, n_vertices)
        elif loss_reduction == 'mean_of_mean':
            self.increments = lambda new_loss, n_vertices : (new_loss/n_vertices, 1)
        else:
            raise ValueError('Unknown loss_reduction parameters {}'.format(loss_reduction))

    def forward(self, outputs):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        loss = 0
        total = 0
        for out in outputs:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(self.device)
            incrs = self.increments(self.loss(out, target), n_vertices)
            loss += incrs[0]
            total += incrs[1]
        return loss/total

def get_criterion(device, loss_reduction):
    return triplet_loss(device, loss_reduction)

# TODO refactor
class loss_mcp(nn.Module):
    def __init__(self, clique_size, device='cpu', loss_reduction='mean', loss=nn.NLLLoss(reduction='mean')):
        super(loss_mcp, self).__init__()
        self.device = device
        self.loss = loss
        self.clique_size = clique_size
        if loss_reduction == 'mean':
            self.increments = lambda new_loss, n_vertices : (new_loss, n_vertices)
        elif loss_reduction == 'mean_of_mean':
            self.increments = lambda new_loss, n_vertices : (new_loss/n_vertices, 1)
        else:
            raise ValueError('Unknown loss_reduction parameters {}'.format(loss_reduction))

    def forward(self, outputs):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        preds = torch.stack((torch.max(outputs[:,:self.clique_size,:],1)[0],
            torch.max(outputs[:,self.clique_size:,:],1)[0]),1)
        log_probas = torch.log_softmax(preds,1)
        target = torch.zeros_like(log_probas[:,0,:],dtype=torch.long)
        target[:,:self.clique_size] = torch.ones((target.shape[0],self.clique_size),dtype=torch.long)
        return self.loss(log_probas,target)

mcp_loss = torch.nn.BCELoss(reduction='none')

class tsp_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='none')):
        super(tsp_loss, self).__init__()
        self.loss = loss
        self.normalize = torch.nn.Sigmoid()#Softmax(dim=2)
        
    def forward(self, raw_scores, mask, target):
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        proba = self.normalize(raw_scores)
        return torch.mean(mask*self.loss(proba,target))