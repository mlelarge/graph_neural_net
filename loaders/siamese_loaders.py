import torch
import torch.nn as nn
import torch.utils

def siamese_loader(data, batch_size , shuffle= True):
    assert len(data) > 0
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4)

