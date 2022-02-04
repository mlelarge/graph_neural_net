import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GatedGraphConv
import dgl
from toolbox.utils import get_device

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GatedGCN(nn.Module):
    def __init__(self,n_layers=10,original_features_num=1,in_features=20, out_features=20, depth_of_mlp=3, **kwargs):
        super().__init__()
        self.conv_start = GatedGraphConv(original_features_num, in_features, 1, 1)
        self.layers = nn.ModuleList()
        for _ in range(n_layers-2):
            layer = GatedGraphConv(in_features, in_features, 1, 1)
            self.layers.append(layer)
        l_layers = []
        for _ in range(depth_of_mlp-1):
            l_layers.append(nn.Linear(in_features,in_features))
            l_layers.append(nn.ReLU())
        l_layers.append(nn.Linear(in_features,out_features))
        l_layers.append(nn.ReLU())
        self.lastmlp = nn.Sequential(*l_layers)
    
    def forward(self,g):
        g = dgl.add_self_loop(g)
        in_feat = g.ndata['feat']
        e_feat = g.edata['feat']
        h = self.conv_start(g, in_feat,e_feat)
        h = F.relu(h)
        for layer in self.layers:
            h = layer(g,h, e_feat)
            h = F.relu(h)
        h = self.lastmlp(h)
        edge_sim = torch.matmul(h,h.T)
        return edge_sim.unsqueeze(0)

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout=0, batch_norm=True, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

class GatedGCNLayerIsotropic(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h)
        #g.update_all(self.message_func,self.reduce_func) 
        g.update_all(fn.copy_u('Bh', 'm'), fn.sum('m', 'sum_h'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_h']
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

class GatedGCNNet_Edge(nn.Module):
    
    def __init__(self, n_layers=4,original_features_num=1,in_features=20,out_features=20, **kwargs):
        super().__init__()
        in_dim = original_features_num
        in_dim_edge = 1 #Only distances
        hidden_dim = in_features
        out_dim = in_features
        n_classes = out_features
        dropout = 0 
        n_layers = n_layers
        self.batch_norm = True #net_params['batch_norm']
        self.residual = True
        self.n_classes = n_classes
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ GatedGCNLayerIsotropic(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
        
    def forward(self, g, h = None, e = None):
        
        if h is None:
            h = g.ndata['feat']
        h = self.embedding_h(h.float())
        if e is None:
            if 'feat' in g.edata:
                e = g.edata['feat']
            else:
                e = torch.ones((g.number_of_edges(),1)).to('cuda')
        e = self.embedding_e(e.float())
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)

        return g.edata['e']
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss

class GatedGCNNet_Node(nn.Module):
    
    def __init__(self, n_layers=8 #12
                ,original_features_num=1,in_features=20,out_features=20, **kwargs):
        super().__init__()
        in_dim = original_features_num
        in_dim_edge = 1 #Only distances
        hidden_dim = in_features
        out_dim = in_features
        n_classes = out_features
        dropout = 0 
        n_layers = n_layers
        self.batch_norm = True #net_params['batch_norm']
        self.residual = True
        self.n_classes = n_classes
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ GatedGCNLayerIsotropic(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
    def forward(self, g, h = None, e = None):
        
        if h is None:
            h = g.ndata['feat']
        h = self.embedding_h(h.float())
        if e is None:
            if 'feat' in g.edata:
                e = g.edata['feat']
            else:
                e = torch.ones((g.number_of_edges(),1)).to('cuda')
        e = self.embedding_e(e.float())
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss