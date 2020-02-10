import torch
import torch.nn as nn

def get_optimizer(args, model):
    optimizer, scheduler = None, None
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=args.lr,
                            amsgrad=False)

    return optimizer, scheduler