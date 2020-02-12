import torch
import torch.nn as nn

class triplet_loss(nn.Module):
    def __init__(self, device='cpu', loss = nn.NLLLoss()):
        super(triplet_loss, self).__init__()
        self.device = device
        self.loss = loss


    def forward(self, out):
        """
        out is the output of siamese network (bs,n_vertices,n_vertices)
        """
        bs = out.shape[0]
        n_vertices = out.shape[1]
        ide = torch.arange(n_vertices)
        target = torch.cat([ide for _ in range(bs)]).to(self.device)
        out_reshape = out.view(-1,n_vertices)
        out_reshape = out_reshape.contiguous()
        return self.loss(out_reshape,target.contiguous())

def get_criterion(device):
    return triplet_loss(device)