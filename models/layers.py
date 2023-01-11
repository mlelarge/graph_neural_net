import torch
import torch.nn as nn
import torch.nn.functional as F
#from maskedtensors.maskedtensor import dispatch_cat
import math
from collections import namedtuple
from torch.nn.parameter import Parameter

# class GraphNorm(nn.Module):
#     def __init__(self, features=0, constant_n_vertices=True, eps =1e-05):
#         super().__init__()
#         self.constant_n_vertices = constant_n_vertices
#         self.eps = eps
    
#     def forward(self, b):
#         return normalize(b, self.constant_n_vertices, self.eps)

class GraphNorm(nn.Module):
    def __init__(self, features, constant_n_vertices=True, elementwise_affine=True, eps =1e-05,device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.constant_n_vertices = constant_n_vertices
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.features = (1,features,1,1)
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.features, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.features, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, b):
        return self.weight*normalize(b, constant_n_vertices=self.constant_n_vertices, eps=self.eps)+self.bias

def normalize(b, constant_n_vertices=True, eps =1e-05):
    means = torch.mean(b, dim = (-1,-2), keepdim=True)
    vars = torch.var(b, unbiased=False,dim = (-1,-2), keepdim=True)
    #(b,f,n1,n2) = b.shape
    #assert n1 == n2
    if constant_n_vertices:
        n = b.size(-1)
    else:
        n = torch.sum(b.mask_dict['N'], dim=1).align_as(vars)
    return (b-means)/(2*torch.sqrt(n*(vars+eps)))


class MlpBlock_Real(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers) except last one
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn = F.relu, constant_n_vertices=True):
        super().__init__()
        self.activation = activation_fn
        self.depth_mlp = depth_of_mlp
        self.cst_vertices = constant_n_vertices
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features
        self.gn = GraphNorm(out_features, constant_n_vertices=constant_n_vertices)

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs[:-1]:
            out = self.activation(conv_layer(out))
        return self.gn(self.convs[-1](out))#normalize(self.convs[-1](out), constant_n_vertices=self.cst_vertices)


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, dim=1)

class Diag(nn.Module):
    def forward(self, xs): return torch.diag_embed(xs)

class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x
    
class Permute(namedtuple('Permute', [])):
    def __call__(self, x): return x.permute(0,2,1)

class Add(nn.Module):
    def forward(self, xs1, xs2): return torch.add(xs1, xs2)

class Matmul(nn.Module):
    def forward(self, xs1, xs2): return torch.matmul(xs1, xs2)

class Matmul_zerodiag(nn.Module):
    def forward(self, xs1, xs2):
        (bs,f,n1,n2) = xs1.shape
        device = xs1.device
        assert n1 == n2
        mask =  torch.ones(n1,n1) - torch.eye(n1,n1)
        mask = mask.reshape((1,1,n1,n1)).to(device)
        mask_b = mask.repeat(bs,f,1,1)
        return torch.matmul(torch.mul(xs1,mask_b), torch.mul(xs2,mask_b))#torch.mul(torch.matmul(xs1, xs2), mask_b)
        #return torch.matmul(zero_diag_fun(xs1), zero_diag_fun(xs2))


def zero_diag_fun(x):
    (bs,f,n1,n2) = x.shape
    device = x.device
    assert n1 == n2
    mask = torch.ones(n1,n1, dtype =x.dtype, device=x.device).fill_diagonal_(0)
    y = x
    for s in x:
        for f in s:
            y *= mask
    return y

class Layernorm(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer_norm = nn.LayerNorm([100,100,n_features], elementwise_affine=False)

    def forward(self, x):
        return self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)

class ColumnMaxPooling(nn.Module):
    """
    take a batch (bs, in_features, n_vertices, n_vertices)
    and returns (bs, in_features, n_vertices)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, -1)[0]

class ColumnSumPooling(nn.Module):
    """
    take a batch (bs, in_features, n_vertices, n_vertices)
    and returns (bs, in_features, n_vertices)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, -1)


class MlpBlock_vec(nn.Module):
    """
    Block of MLP layers acting on vectors (bs, features, n)
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn = F.relu):
        super().__init__()
        self.activation = activation_fn
        self.depth_mlp = depth_of_mlp
        self.mlp = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.mlp.append(nn.Linear(in_features, out_features))
            _init_weights(self.mlp[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs.permute(0,2,1)
        for fc in self.mlp[:-1]:
            out = self.activation(fc(out))
        return self.mlp[-1](out).permute(0,2,1)

class SelfAttentionLayer(nn.Module):
    def __init__(self, config, dmt=False):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.emb_hea = self.n_embd//self.n_heads
        #print(self.emb_hea)
        self.dmt = dmt
        self.Query = nn.Linear(self.n_embd, self.n_embd)
        self.Key = nn.Linear(self.n_embd, self.n_embd)
        self.Value = nn.Linear(self.n_embd, self.n_embd)
    
    def forward(self, x): # x (bs, T, ne)
        b,t,n = x.size()
        #x = x.permute(0,2,1)
        Q = self.Query(x) # (bs, T, ne)
        Q = Q.view(b,t,self.n_heads,self.emb_hea).permute(0,2,1,3) # (bs, nh, T, nne)
        if self.dmt:
            Q.tensor.rename_(N='N_')
        K = torch.div(self.Key(x),math.sqrt(self.emb_hea)) # (bs, T, ne)
        K = K.view(b,t,self.n_heads,self.emb_hea).permute(0,2,1,3) # (bs, nh, T, nne)
        V = self.Value(x) # (bs, T, ne)
        V = V.view(b,t,self.n_heads,self.emb_hea).permute(0,2,1,3) # (bs, nh, T, nne)
        #A = torch.einsum('ntk,nsk->nst', Q, K) # (bs, T, kc), (bs, T, kc) -> (bs , T, T)
        A = torch.matmul(K, Q.permute(0,1,3,2)) #(bs, nh, T, T)
        A = F.softmax(A, dim=-1)
        y = torch.matmul(A, V)#torch.bmm(A, V) # (bs, nh, T, nne)
        y = y.permute(0,2,1,3).contiguous().view(b, t, n) # (bs, T, ne)
        return y, A

class AttentionBlock_vec(nn.Module):
    """
    Attention Block of MLP layers acting on vectors (bs, features, n)
    """
    def __init__(self, nb_features, depth_of_mlp, nb_heads=1 ,dmt=True, activation_fn = F.relu):
        super().__init__()
        self.activation = activation_fn
        self.depth_mlp = depth_of_mlp
        self.mlp = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.mlp.append(nn.Linear(nb_features, nb_features))
            _init_weights(self.mlp[-1])
            #in_features = out_features
                            
        class config:
            n_embd = nb_features
            n_heads = nb_heads
        self.attn = SelfAttentionLayer(config, dmt)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def mlpf(self, x):
        out = x
        for fc in self.mlp[:-1]:
            out = self.activation(fc(out))
        return self.mlp[-1](out)    
        
    def forward(self, x):
        y, A = self.attn(self.ln_1(x))
        x = torch.add(x, y)
        return torch.add(x,self.mlpf(self.ln_2(x)))

class GraphAttentionLayer(nn.Module):
    def __init__(self, config, dmt=False):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.depth_of_mlp = config.d_of_mlp
        self.emb_hea = self.n_embd//self.n_heads
        #print(self.emb_hea)
        self.dmt = dmt
        self.Query = nn.Linear(self.n_embd, self.n_embd)
        self.Key = nn.Linear(self.n_embd, self.n_embd)
        self.Value = MlpBlock_Real(self.n_embd,self.n_embd,self.depth_of_mlp)#nn.Linear(self.n_embd, self.n_embd)
    
    def forward(self, x): # x (bs, ne, T, T)
        b,n,t,t = x.size()
        V = self.Value(x) # (bs, ne, T, T)
        x = x.permute(0,2,3,1) # (bs, T, T, ne)
        Q = normalize(self.Query(x)) # (bs, ne, T, T)
        K = normalize(self.Key(x)) # (bs, ne, T, T)
        Q = Q.view(b,t,t,self.n_heads,self.emb_hea).permute(0,3,4,1,2) # (bs, nh, nne, T, T)
        K = K.view(b,t,t,self.n_heads,self.emb_hea).permute(0,3,4,1,2) # (bs, nh, nne, T, T)
        V = V.view(b,self.n_heads,self.emb_hea,t,t)#.permute(0,3,4,1,2) # (bs, nh, nne, T, T)
        A = torch.einsum('nhftu,nhfru->nhtr', Q, K) # (bs,nh,nne,T, T), (bs,nh,nne,T,T) -> (bs,nh, T, T)
        A = F.softmax(A, dim=-1)
        y = torch.einsum('bhst,bhfst -> bhfs', A, V) # (bs, nh,nne T, T)
        #y = torch.matmul(A.unsqueeze(2), V) # (bs, nh,nne T, T)
        y = y.contiguous().view(b, n, t)
        return y#normalize(y)

class GraphAttentionLayer_mlp(nn.Module):
    def __init__(self, config, dmt=False):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.depth_of_mlp = config.d_of_mlp
        self.emb_hea = self.n_embd//self.n_heads
        #print(self.emb_hea)
        self.dmt = dmt
        self.Query = MlpBlock_Real(self.n_embd,self.n_embd,self.depth_of_mlp)#nn.Linear(self.n_embd, self.n_embd)
        self.Key = MlpBlock_Real(self.n_embd,self.n_embd,self.depth_of_mlp)#nn.Linear(self.n_embd, self.n_embd)
        self.Value = MlpBlock_Real(self.n_embd,self.n_embd,self.depth_of_mlp)#nn.Linear(self.n_embd, self.n_embd)
    
    def forward(self, x): # x (bs, ne, T, T)
        b,n,t,t = x.size()
        V = self.Value(x) # (bs, ne, T, T)
        #print(x.shape)
        #x = x.permute(0,2,3,1) # (bs, T, T, ne)
        Q = self.Query(x) # (bs,ne, T, T)
        Q = Q.view(b,self.n_heads,self.emb_hea,t,t) # (bs, nh, nne, T, T)
        K = self.Key(x)#torch.div(self.Key(x),math.sqrt(self.emb_hea)) # (bs,ne,  T, T)
        K = K.view(b,self.n_heads,self.emb_hea,t,t) # (bs, nh, nne, T, T)
        V = V.view(b,self.n_heads,self.emb_hea,t,t) # (bs, nh, nne, T, T)
        A = torch.einsum('nhftu,nhfru->nhtr', Q, K) # (bs,nh,nne,T, T), (bs,nh,nne,T,T) -> (bs,nh, T, T)
        A = F.softmax(A, dim=-1)
        #print(A.unsqueeze(2).shape)
        #print(V.shape)
        #y = torch.matmul(A.unsqueeze(2), V) # (bs, nh,nne T, T)
        y = torch.einsum('bhst,bhfst -> bhfst', A, V) # (bs, nh,nne T, T)
        y = y.contiguous().view(b, n, t, t)
        return normalize(y)