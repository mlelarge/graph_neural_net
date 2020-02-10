import torch
import torch.nn as nn

def triplet_loss(out, device='cpu', loss = nn.NLLLoss()):
    """
    out is the output of siamese network (bs,n_vertices,n_vertices)
    """
    bs = out.shape[0]
    n_vertices = out.shape[1]
    ide = torch.arange(n_vertices)
    target = torch.cat([ide for _ in range(bs)]).to(device)
    out_reshape = out.view(-1,n_vertices)
    return loss(out_reshape,target)