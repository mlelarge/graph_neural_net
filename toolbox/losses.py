import torch
import torch.nn as nn

class triplet_loss(nn.Module):
    def __init__(self, device='cpu', loss = nn.NLLLoss()):
        super(triplet_loss, self).__init__()
        self.device = device
        self.loss = loss


    def forward(self, outputs):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        for out in outputs:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(self.device)
            return self.loss(out, target)

def get_criterion(device):
    return triplet_loss(device)
