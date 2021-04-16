import torch.nn as nn

class Verbose_Model(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.block_save = {}

        for block in self.model.base_model.reg_blocks:
            block.register_forward_hook(lambda block, input, output: self.block_update(block.name,output)) #Will save the output of each block
    
    def block_update(self,name,value):
        self.block_save[name] = value
    
    def forward(self,x):
        return self.model.forward(x)