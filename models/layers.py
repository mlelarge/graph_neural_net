import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x n_vertices x n_vertices
    Take the input through 2 parallel MLP routes, multiply the result 
    and pass the concatenation of this result with the input through a MLP
    with no activation for the last layer.
    """
    def __init__(self, in_features, out_features, depth_of_mlp):
        super().__init__()

        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp3 = MlpBlock(in_features+out_features, out_features,depth_of_mlp)
        self.last_layer = nn.Conv2d(out_features,out_features,kernel_size=1, padding=0, bias=True)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)
        mult = torch.matmul(mlp1, mlp2)
        out = torch.cat((inputs, mult), dim=1)
        out = self.mlp3(out)
        out = self.last_layer(out)
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn = F.relu):
        super().__init__()
        self.activation = activation_fn
        self.depth_mlp = depth_of_mlp
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
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

class Features_2_to_1(nn.Module):
    def __init__(self):
        """
        take a batch (bs, n_vertices, n_vertices, in_features)
        and returns (bs, n_vertices, basis * in_features)
        where basis = 5
        """
        super().__init__()

    def forward(self, x):
        b, n, _, in_features = x.size()
        basis = 5
        diag_part = torch.diagonal(x, dim1=1, dim2=2).permute(0,2,1)
        max_diag_part = torch.max(diag_part, 1)[0].unsqueeze(1)
        max_of_rows = torch.max(x, 2)[0]
        max_of_cols = torch.max(x, 1)[0]
        max_all = torch.max(torch.max(x, 1)[0], 1)[0].unsqueeze(1)

        op1 = diag_part
        op2 = max_diag_part.expand_as(op1)
        op3 = max_of_rows
        op4 = max_of_cols
        op5 = max_all.expand_as(op1)

        output = torch.stack([op1, op2, op3, op4, op5], dim=2)
        assert output.size() == (b, n, basis, in_features), output.size()
        return output.view(b, n, basis*in_features)

class ColumnMaxPooling(nn.Module):
    """
    take a batch (bs, n_vertices, n_vertices, in_features)
    and returns (bs, n_vertices, in_features)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, 2)[0]
