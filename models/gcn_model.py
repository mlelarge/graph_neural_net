from networkx.algorithms.assortativity import connectivity
import torch
import torch.nn as nn
from torch.functional import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import networkx as nx
#from numpy import indices as npindices, argpartition as npargpartition, array as nparray
import dgl
from dgl.nn import GraphConv

import tqdm

from typing import Tuple

#from loaders.data_generator import TSP_Generator


# def data_to_dgl_format(data_object,problem=None,**kwargs):
#     return DGL_Loader.from_data_generator(data_object,problem,**kwargs)

# class DGL_Loader(torch.utils.data.Dataset):
#     def __init__(self):
#         self.data = []
    
#     @staticmethod
#     def from_data_generator(data_object, problem, **kwargs):
#         loader = DGL_Loader()
#         print("Converting data to DGL format")
        
#         for data,target in tqdm.tqdm(data_object.data):
#             elt_dgl = connectivity_to_dgl(data)
#             loader.data.append((elt_dgl,target))
#         print("Conversion ended.")
#         return loader
    
#     def __getitem__(self, i):
#         """ Fetch sample at index i """
#         return self.data[i]

#     def __len__(self):
#         """ Get dataset length """
#         return len(self.data)


class SimpleGCN(nn.Module):
    def __init__(self, original_features_num, in_features, out_features, **kwargs):
        super(SimpleGCN, self).__init__()
        self.conv1 = GraphConv(original_features_num, in_features)
        self.conv2 = GraphConv(in_features, out_features)

    def forward(self, g):
        g = dgl.add_self_loop(g)
        in_feat = g.ndata['feat']
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h.unsqueeze(0)

class BaseGCN(nn.Module):
    def __init__(self,n_layers=20,original_features_num=1,in_features=20,out_features=20, **kwargs):
        super().__init__()
        self.conv_start = GraphConv(original_features_num, in_features)
        self.layers = nn.ModuleList()
        for _ in range(n_layers-2):
            layer = GraphConv(in_features, in_features)
            self.layers.append(layer)
        self.conv_final = GraphConv(in_features, out_features)
    
    def forward(self,g):
        g = dgl.add_self_loop(g)
        in_feat = g.ndata['feat']
        h = self.conv_start(g, in_feat)
        h = F.relu(h)
        for layer in self.layers:
            h = layer(g,h)
            h = F.relu(h)
        h = self.conv_final(g, h)
        return h.unsqueeze(0)

# if __name__=="__main__":
#     print("Main executing")
#     from loaders.data_generator import generate_erdos_renyi_netx, adjacency_matrix_to_tensor_representation
#     N = 50
#     p = 0.2
#     g, W = generate_erdos_renyi_netx(p,N)
#     connect = adjacency_matrix_to_tensor_representation(W)
#     dglgraph = _connectivity_to_dgl_adj(connect)
#     connect_back = _dgl_adj_to_connectivity(dglgraph)
#     print(torch.all(connect==connect_back))