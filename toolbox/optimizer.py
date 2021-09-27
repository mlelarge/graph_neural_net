import torch
import torch.nn as nn

def get_optimizer(args, model):
    optimizer, scheduler = None, None
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=args['lr'],
                            amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = args['scheduler_decay'],patience=args['scheduler_step'])#2)#StepLR(optimizer, step_size=args['scheduler_step'], gamma=args['scheduler_decay'])

    return optimizer, scheduler
