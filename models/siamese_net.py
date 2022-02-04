from typing import Tuple
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import Simple_Node_Embedding

from models.gated_gcn import GatedGCNNet_Node, GatedGCNNet_Edge

class Siamese_Model(nn.Module):
    def __init__(self, original_features_num, num_blocks, in_features,out_features, depth_of_mlp, embedding_class=Simple_Node_Embedding):
        """
        take a batch of pair of graphs 
        ((bs, n_vertices, n_vertices, in_features) (bs,n_vertices, n_vertices, in_features))
        and return a batch of node similarities (bs, n_vertices, n_vertices)
        for each node the sum over the second dim should be one: sum(torch.exp(out[b,i,:]))==1
        graphs must have same size inside the batch
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp = depth_of_mlp
        self.node_embedder = embedding_class(original_features_num, num_blocks, in_features,out_features, depth_of_mlp)

    def forward(self, x):
        """
        Data should be given with the shape (x1,x2)
        """
        assert x.shape[1]==2, f"Data given is not of the shape (x1,x2) => data.shape={x.shape}"
        x = x.permute(1,0,2,3,4)
        x1 = x[0]
        x2 = x[1]
        x1 = self.node_embedder(x1)
        x2 = self.node_embedder(x2)
        raw_scores = torch.matmul(x1,torch.transpose(x2, 1, 2))
        return raw_scores

class Siamese_Model_Gen(nn.Module):
    def __init__(self, Model_class,**kwargs):
        """
        General class enforcing a Siamese architecture.
        The forward usually takes in a pair of graphs of shape:
        ((bs, n_vertices, n_vertices, in_features) (bs, n_vertices, n_vertices, in_features))
        and return a batch of node similarities (bs, n_vertices, n_vertices).
        That was the base use, but model can be anything and return mostly anything as long as the helper is taken into account
        """
        self.model=''
        super().__init__()
        self.node_embedder = Model_class(**kwargs)
        if isinstance(self.node_embedder,GatedGCNNet_Node):
            self.model='gatedgcnnode'
        elif isinstance(self.node_embedder,GatedGCNNet_Edge):
            self.model='gatedgcnedge'

    def forward(self,x):
        """
        Data should be given with the shape (x1,x2)
        """
        if isinstance(x,torch.Tensor):
            assert x.shape[1]==2, f"Data given is not of the shape (x1,x2) => data.shape={x.shape}"
            x = x.permute(1,0,2,3,4)
            x1 = x[0]
            x2 = x[1]
        else:
            assert len(x)==2, f"Data given is not of the shape (x1,x2) => data.shape={x.shape}"
            x1 = x[0]
            x2 = x[1]
        x1_out = self.node_embedder(x1)
        x2_out = self.node_embedder(x2)
        if self.model=='gatedgcnedge':
            #We're in dgl territory
            N = x1.number_of_nodes()
            N_features = x1_out.shape[1]
            src1,dst1 = x1.edges(form='uv')
            src2,dst2 = x2.edges(form='uv')
            assert src1.shape[0] == x1_out.shape[0] and src2.shape[0] == x2_out.shape[0]
            assert x1.number_of_nodes()==x2.number_of_nodes()
            fullt = torch.zeros((N,N),dtype=bool,requires_grad=False)
            fullt1 = torch.zeros((N,N,N_features))
            fullt2 = torch.zeros_like(fullt1)
            for i,(in1,out1,in2,out2) in enumerate(zip(src1,dst1,src2,dst2)):
                fullt[in1,out1] = True
                fullt[in2,out2] = True
                fullt1[in1,out1] = x1_out[i]
                fullt2[in2,out2] = x2_out[i]
            raw_scores = torch.matmul(fullt1,torch.transpose(fullt2,1,2)).unsqueeze(0) #unsqueeze to imitate a batch of 1
        elif self.model=='gatedgcnnode':
            raw_scores = torch.matmul(x1_out,torch.transpose(x2_out,-2,-1)).unsqueeze(0)
        else:
            raw_scores = torch.matmul(x1_out,torch.transpose(x2_out, 1, 2))
        return raw_scores