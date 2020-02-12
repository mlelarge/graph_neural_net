import torch
import torch.nn as nn

class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """
    def __init__(self, in_features, out_features, depth_of_mlp):
        super().__init__()

        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp3 = MlpBlock(in_features+out_features, out_features, depth_of_mlp)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)
        mult = torch.matmul(mlp1, mlp2)
        out = torch.cat((inputs, mult), dim=1)
        out = self.mlp3(out)
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    #if layer.bias is not None:
    #    nn.init.zeros_(layer.bias)

##### END OF CODE FROM github.com/hadarser/ProvablyPowerfulGraphNetworks_torch #####

class MlpBlock1d(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, list_of_features, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for (i, in_feature) in enumerate(list_of_features[:-1]):
            self.convs.append(nn.Conv1d(in_feature, list_of_features[i+1], kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs[:-1]:
            out = self.activation(conv_layer(out))
        out = self.convs[-1](out)

        return out
