import torch
import torch.nn as nn

class triplet_loss(nn.Module):
    def __init__(self, device='cpu', loss = nn.NLLLoss(reduction='sum')):
        super(triplet_loss, self).__init__()
        self.device = device
        self.loss = loss


    def forward(self, outputs):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        loss = 0
        total_n_vertices = 0
        for out in outputs:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(self.device)
            loss += self.loss(out, target)
            total_n_vertices += n_vertices
        return loss/total_n_vertices

def get_criterion(device):
    return triplet_loss(device)
