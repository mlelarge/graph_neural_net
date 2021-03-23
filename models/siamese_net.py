import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import Simple_Node_Embedding

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
        x.permute(1,0,2,3,4)
        x1 = x[0]
        x2 = x[1]
        x1 = self.node_embedder(x1)
        x2 = self.node_embedder(x2)
        raw_scores = torch.matmul(x1,torch.transpose(x2, 1, 2))
        return raw_scores
