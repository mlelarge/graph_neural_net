import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.utils import get_device
import itertools


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

    def forward(self, outputs, target_dummy):#Keep target for modularity, target should be an empty tensor, but it's not used anyways
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
        raw_scores =  raw_scores * raw_scores.transpose(-2,-1) #Symmetrization before proba
        #raw_scores *= 1-torch.eyer(raw_scores.shape[-1])
        proba = self.normalize(raw_scores)
        #print(proba)
        #proba = proba*proba.transpose(-2,-1) #Symmetrization
        loss = torch.sum(torch.sum(proba*distance_matrix, dim=-1), dim=-1) #In need of normalization of the loss
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
    def __init__(self, loss=nn.MSELoss(reduction='none'), normalize = nn.Sigmoid()):
        super(sbm_loss, self).__init__()
        self.loss = loss
        self.normalize = normalize
    
    def fake_forward(self, embeddings, target):
        """
        embeddings (bs,n_vertices,dim_embedding)
        cluster_sizes (bs, n_clusters)
        """
        device = get_device(embeddings)
        bs,n,_ = embeddings.shape
        cluster_sizes = ((n//2)*torch.ones((bs,2))).to(int)
        loss = torch.zeros([1], dtype=torch.float64, device=device)
        for i, n_nodes in enumerate(cluster_sizes):
            mean_cluster = []
            var_cluster = []
            prev = 0
            for n in n_nodes:
                mean_cluster.append(
                    F.normalize(torch.mean(embeddings[i][prev : prev + n, :], 0), dim=0)
                )
                var_cluster.append(torch.var(embeddings[i][prev : prev + n, :], 0))
                prev = n

            loss_dist_cluster = torch.zeros([1], dtype=torch.float64, device=device)
            loss_var_cluster = torch.zeros([1], dtype=torch.float64, device=device)
            for m1, m2 in itertools.combinations(mean_cluster, 2):
                loss_dist_cluster += 1.0 + torch.dot(m1, m2)

            mu = 1.0
            loss_var_cluster = mu * torch.stack(var_cluster, dim=0).sum()

            loss += 0.1 * loss_dist_cluster + loss_var_cluster
        return loss
    
    
    def forward(self, raw_scores, target):
        """
        raw_scores of shape (bs,n_vertices,out_features), target of shape (bs,n_vertices,n_vertices)
        """
        embeddings = F.normalize(raw_scores) #Compute E
        similarity = embeddings @ embeddings.transpose(-2,-1) #Similarity = E@E.T
        loss = self.loss(similarity,target)
        return torch.mean(loss)
    

