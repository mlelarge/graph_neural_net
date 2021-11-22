from networkx.algorithms.assortativity import connectivity
import torch
import torch.nn as nn
from torch.functional import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import networkx as nx
from numpy import indices as npindices
import dgl
from dgl.nn import GraphConv
from numpy import mgrid as npmgrid
import tqdm

from typing import Tuple

ADJ_UNIQUE_TENSOR = torch.Tensor([0.,1.])

def is_adj(matrix):
    return torch.all((matrix==0) + (matrix==1))

def _connectivity_to_dgl_adj(connectivity):
    assert len(connectivity.shape)==3, "Should have a shape of N,N,2"
    adj = connectivity[:,:,1] #Keep only the adjacency (discard node degree)
    N,_ = adj.shape
    assert is_adj(adj), "This is not an adjacency matrix"
    mgrid = npmgrid[:N,:N].transpose(1,2,0)
    edges = mgrid[torch.where(adj==1)]
    edges = edges.T #To have the shape (2,n_edges)
    src,dst = [elt for elt in edges[0]], [elt for elt in edges[1]] #DGLGraphs don't like Tensors as inputs...
    gdgl = dgl.graph((src,dst),num_nodes=N)
    gdgl.ndata['feat'] = connectivity[:,:,0].diagonal().reshape((N,1)) #Keep only degree
    return gdgl

def _dgl_adj_to_connectivity(dglgraph):
    N = len(dglgraph.nodes())
    connectivity = torch.zeros((N,N,2))
    edges = dglgraph.edges()
    for i in range(dglgraph.num_edges()):
        connectivity[edges[0][i],edges[1][i],1] = 1
    degrees = connectivity[:,:,1].sum(1)
    indices = torch.arange(N)
    print(degrees.shape)
    connectivity[indices, indices, 0] = degrees
    return connectivity

def _connectivity_to_dgl_edge(connectivity,sparsify=None):
    """Converts a connectivity tensor to a dgl graph with edge and node features.
    if 'sparsify' is specified, it should be an integer : the number of closest nodes to keep
    """
    assert len(connectivity.shape)==3, "Should have a shape of N,N,2"
    N,_,_ = connectivity.shape
    distances = connectivity[:,:,1]
    mask = torch.ones((N,N))
    if sparsify is not None:
        pass
    connectivity = connectivity*mask
    adjacency = (connectivity!=0).to(torch.float)
    gdgl = _connectivity_to_dgl_adj(adjacency)
    src,rst = gdgl.edges() #For now only contains node features
    gdgl.edata["feat"] = connectivity[src,rst]
    return gdgl


def connectivity_to_dgl(connectivity_graph):
    """Converts a simple connectivity graph (with weights on edges if needed) to a pytorch-geometric data format"""
    if len(connectivity_graph.shape)==4:#We assume it's a siamese dataset, thus of shape (2,N,N,in_features)
        assert connectivity_graph.shape[0]==2
        assert connectivity_graph.shape[1]==connectivity_graph.shape[2]
        graph1,graph2 = connectivity_to_dgl(connectivity_graph[0]), connectivity_to_dgl(connectivity_graph[1])
        return (graph1,graph2)
    elif len(connectivity_graph.shape)==3:#We assume it's a simple dataset, thus of shape (N,N,in_features)
        assert connectivity_graph.shape[0]==connectivity_graph.shape[1]
        if is_adj(connectivity_graph[:,:,1]):
            return _connectivity_to_dgl_adj(connectivity_graph)
        return _connectivity_to_dgl_edge(connectivity_graph)

def data_to_dgl_format(data_object):
    return DGL_Loader.from_data_generator(data_object)

class DGL_Loader(torch.utils.data.Dataset):
    def __init__(self):
        self.data = []
    
    @staticmethod
    def from_data_generator(data_object):
        loader = DGL_Loader()
        print("Converting data to DGL format")
        for data,target in tqdm.tqdm(data_object.data):
            elt_dgl = connectivity_to_dgl(data)
            loader.data.append((elt_dgl,target))
        print("Conversion ended.")
        return loader
    
    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)

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

if __name__=="__main__":
    print("Main executing")
    from loaders.data_generator import generate_erdos_renyi_netx, adjacency_matrix_to_tensor_representation
    N = 50
    p = 0.2
    g, W = generate_erdos_renyi_netx(p,N)
    connect = adjacency_matrix_to_tensor_representation(W)
    dglgraph = _connectivity_to_dgl_adj(connect)
    connect_back = _dgl_adj_to_connectivity(dglgraph)
    print(torch.all(connect==connect_back))