import torch
import torch.nn as nn
from models.layers import RegularBlock


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.config = config
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = config.node_labels + 1  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(last_layer_features, next_layer_features, config.architecture.depth_of_mlp)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

##### END OF CODE FROM github.com/hadarser/ProvablyPowerfulGraphNetworks_torch #####
        self.classification = config.classification
        self.pretrained_classification = config.pretrained_classification
                
        if config.classification:
            self.suffix = suffix.MaxSuffixClassification()
            if not config.pretrained_classification:
                self.fc_layers = nn.ModuleList()
                self.fc_layers.append(modules.FullyConnected(2*block_features[-1], 512))
                self.fc_layers.append(modules.FullyConnected(512, 256))
                self.fc_layers.append(modules.FullyConnected(256, 2, activation_fn=None))
        else:
            self.suffix = suffix.Features_2_to_1()
            #self.mlp = modules.MlpBlock1d([5*block_features[-1], 512, 256, 128, 64])

    def forward(self, x):
        #here x.shape = (bs, n_vertices, n_vertices, n_features)
        x = x.permute(0, 3, 1, 2)
        #expects x.shape = (bs, n_features, n_vertices, n_vertices)
        for block in self.reg_blocks:
            x = block(x)
        x = self.suffix(x)
        if not self.classification:
            assert len(x.size()) == 3
            # here x.shape = (bs, n_features, n_vertices)
            x = x.permute(0, 2, 1)
            # here x.shape = (bs,n_vertices, n_features)
        else:
            if not self.pretrained_classification:
                for l in self.fc_layers:
                    x = l(x)
        return x
