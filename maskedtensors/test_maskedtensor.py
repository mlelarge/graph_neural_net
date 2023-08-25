"""
Tests for maskedtensor module
To execute, run python -m pytest at the root of the project

Recommanded: install pytest-repeat to repeat tests with e.g.
    python -m pytest . --count 10
"""

import functools
import pytest
import torch
import torch.nn as nn
import maskedtensor
#from models.graph_classif import Graph_Classif
#from models.layers import MlpBlock, RegularBlock, MlpBlock_Real, Scaled_Block, MlpBlock_vec
from models.layers import MlpBlock_Real, MlpBlock_vec, Matmul, normalize, GraphNorm, Concat, Add, Diag
#from models.base_model_old import Node_Embedding, Graph_Embedding
#from models.siamese_net import Siamese_Node
from toolbox.metrics import accuracy_linear_assignment, accuracy_max
from toolbox.losses import triplet_loss

def apply_list_tensors(lst, func):
    """ Apply func on each tensor (with batch dim) """
    batched_lst = [tens.unsqueeze(0) for tens in lst]
    batched_res_lst = [func(tens) for tens in batched_lst]
    res_lst = [tens.squeeze(0) for tens in batched_res_lst]
    return res_lst

def apply_binary_list_tensors(lst, func):
    """ Apply func on each tensor (with batch dim) """
    batched_lst = [(tens.unsqueeze(0), other.unsqueeze(0)) for tens, other in lst]
    batched_res_lst = [func(*tpl) for tpl in batched_lst]
    res_lst = [tens.squeeze(0) for tens in batched_res_lst]
    return res_lst

N_FEATURES = 16
N_VERTICES_RANGE = range(40,50)
FIXED_N_VERTICES = 50
ATOL = 1e-5
DEVICE = torch.device('cpu')

@pytest.fixture
def tensor_list():
    """ Generate list of tensors (graphs)"""
    lst = [torch.empty((N_FEATURES, n_vertices, n_vertices)).normal_()
           for n_vertices in N_VERTICES_RANGE]
    return lst

@pytest.fixture
def tensor_listvec():
    """ Generate list of tensors (vector) """
    lst = [torch.empty((N_FEATURES, n_vertices)).normal_()
           for n_vertices in N_VERTICES_RANGE]
    return lst

@pytest.fixture
def score_list():
    """ Generate list of tensors with no features and fixed n_vertices"""
    lst = [torch.empty((FIXED_N_VERTICES, FIXED_N_VERTICES)).normal_()
           for _ in N_VERTICES_RANGE]
    return lst

other_tensor_list = tensor_list

""" def graph_conv_wrapper(func):
    Applies graph convention to a function using pytorch convention
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        new_args = [x.permute(0, 3, 1, 2) for x in args]
        ret = func(*new_args, **kwargs)
        return ret.permute(0, 2, 3, 1)
    return wrapped_func
 """

def accuracy_wrapper(func):
    """ Wraps accuracy funcs so that they behave like the other funcs """
    @functools.wraps(func)
    def wrapped_func(weights, *args, **kwargs):
        # remove features
        new_weights = torch.sum(weights, -1)
        ret = func(new_weights, *args, **kwargs)
        return torch.Tensor(ret)
    return wrapped_func

# the third parameter specifies whether the base_name of the second maskedtensor
# should match the first one
TEST_BINARY_FUNCS = [
        # when pytorch issue is fixed, change this
        #(lambda t1, t2: maskedtensor.dispatch_cat((t1, t2), dim=-1), 'torch.cat', True),
        (lambda t1, t2: torch.cat((t1, t2), dim=1), 'torch.cat', True),
        (lambda t1, t2: torch.stack((t1, t2), dim=1), 'torch.stack', True),
        (torch.matmul, 'torch.matmul', True),
        (torch.matmul, 'torch.matmul', False),
        #(Siamese_Node(N_FEATURES, 2, 32, 32, 3), 'Siamese_Node', False),
        (Matmul(), 'Matmul', False),
        (Concat(), 'Concat', True),
        (Add(), 'Add', True)]
# embedding is not working yet...

@pytest.mark.parametrize('func_data', TEST_BINARY_FUNCS, ids=lambda func_data: func_data[1])
def test_binary_torch_func(tensor_list, other_tensor_list, func_data):
    """ Test torch function wich use two tensors """
    func, _, same_base_name  = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(1, 2))
    other_base_name = 'N' if same_base_name else 'M'
    other_masked_tensor = maskedtensor.from_list(other_tensor_list, dims=(1, 2),
            base_name=other_base_name)
    res_mt = list(func(masked_tensor, other_masked_tensor))
    binary_list = zip(tensor_list, other_tensor_list)
    res_lst = apply_binary_list_tensors(binary_list, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.allclose(t_mt, t_lst, atol=ATOL), torch.norm(t_mt - t_lst, p=float('inf'))

ln = nn.LayerNorm(N_FEATURES)

TEST_FUNCS = [
    (lambda t: torch.add(t, 1), 'torch.add'),
    (lambda t: torch.mul(t, 2), 'torch.mul'),
    (lambda t: torch.sum(t, 2), 'torch.sum'),
    (lambda t: torch.max(t, 2)[0], 'torch.max(dim=2)'),
    (lambda t: torch.mean(t, dim=(-2,-1)), 'torch.mean'),
    (lambda t: torch.var(t, unbiased=False, dim=(-2,-1)), 'torch.var'),
    # keep first dim not to perturb apply_list_tensors
    (lambda t: t.permute(0, 3, 2, 1), 'permute'),
    (nn.Conv2d(N_FEATURES, 2*N_FEATURES, 1), 'nn.Conv2d'),
    (lambda t: ln(t.permute(0,3,2,1)), 'nn.LayerNorm'),
    (nn.InstanceNorm2d(N_FEATURES, affine=False, track_running_stats=False), 'InstanceNorm2d'),
    (nn.InstanceNorm2d(N_FEATURES, affine=True, track_running_stats=False), 'InstanceNorm2d_affine'),
    (lambda t: torch.diag_embed(t,dim1=-2,dim2=-1), 'torch.diag_embed'),
    (Diag(), 'Diag')
    #(MlpBlock(N_FEATURES, 2*N_FEATURES, 2), 'MlpBlock'),
    #(RegularBlock(N_FEATURES, 2*N_FEATURES, 2), 'RegularBlock'),
    #(Node_Embedding(N_FEATURES, 2, 32, 32, 3), 'Simple_Node_Embedding'),
    #(MlpBlock_Real(N_FEATURES, 2*N_FEATURES, 3), 'MlpBlock_Real'),
    #(Scaled_Block(N_FEATURES, 2*N_FEATURES, 2), 'Scaled_Block'),
    #(Graph_Embedding(N_FEATURES, 2, 32, 32, 3), 'Graph_Embedding'),
    #(Graph_Classif(N_FEATURES, 2, 32, 32, 3), 'Graph_Classif')
    ]

@pytest.mark.parametrize('func_data', TEST_FUNCS, ids=lambda func_data: func_data[1])
def test_torch_func(tensor_list, func_data):
    """ Test torch function """
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(1, 2))
    res_mt = list(func(masked_tensor))
    res_lst = apply_list_tensors(tensor_list, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.allclose(t_mt, t_lst, atol=ATOL), torch.norm(t_mt - t_lst, p=float('inf'))

TEST_CUST_FUNCS = [
    (normalize, 'normalize')
    ]

@pytest.mark.parametrize('func_data', TEST_CUST_FUNCS, ids=lambda func_data: func_data[1])
def test_custom_func(tensor_list, func_data):
    """ Test custom function """
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(1, 2))
    res_mt = list(func(masked_tensor, constant_n_vertices=False))
    res_lst = apply_list_tensors(tensor_list, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.allclose(t_mt, t_lst, atol=ATOL), torch.norm(t_mt - t_lst, p=float('inf'))

mlp_mt = MlpBlock_Real(N_FEATURES, 2*N_FEATURES, 2, constant_n_vertices=False)
mlp = MlpBlock_Real(N_FEATURES, 2*N_FEATURES, 2)
mlp.convs = mlp_mt.convs
gn_mt = GraphNorm(N_FEATURES, constant_n_vertices=False)
gn = GraphNorm(N_FEATURES)

TEST_LAYERS = [
    (mlp_mt, mlp, 'MlpBlock_Real'),
    (gn_mt, gn, 'GraphNorm')
]

@pytest.mark.parametrize('func_data', TEST_LAYERS, 
    ids=lambda func_data: func_data[2])
def test_layers(tensor_list, func_data):
    """ Test layer """
    func_mt, func, _ = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(1, 2))
    res_mt = list(func_mt(masked_tensor))
    res_lst = apply_list_tensors(tensor_list, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.allclose(t_mt, t_lst, atol=ATOL), torch.norm(t_mt - t_lst, p=float('inf'))


TEST_MAX =[
    (lambda t: torch.max(t), 'torch.max')
]

@pytest.mark.parametrize('fun_max', TEST_MAX, ids=lambda fun_max: fun_max[1])
def test_max(tensor_list, fun_max):
    f_max , _ = fun_max 
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(1, 2))
    res_mt = f_max(masked_tensor)
    res_lst = torch.max(torch.tensor(apply_list_tensors(tensor_list, f_max)))
    assert torch.allclose(res_mt, res_lst, atol=ATOL)

TEST_VEC = [
    (lambda t: torch.max(t, dim=1)[0], 'torch.max_vec'),
    (MlpBlock_vec(N_FEATURES, 2*N_FEATURES, 2), 'MlpBlock_vec'),
    (lambda t: ln(t.permute(0,2,1)), 'nn.LayerNorm_vec')
]

@pytest.mark.parametrize('func_data', TEST_VEC, ids=lambda func_data: func_data[1])
def test_vec_func(tensor_listvec, func_data):
    """ Test vec function """
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(tensor_listvec, dims=(1,))
    res_mt = list(func(masked_tensor))
    res_lst = apply_list_tensors(tensor_listvec, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.allclose(t_mt, t_lst, atol=ATOL), torch.norm(t_mt - t_lst, p=float('inf'))


TEST_LOSS_FUNCS = [
        (triplet_loss(loss_reduction='mean'), 'loss_mean'),
        (triplet_loss(loss_reduction='mean_of_mean'), 'loss_mean_of_mean')]

@pytest.mark.parametrize('func_data', TEST_LOSS_FUNCS, ids=lambda func_data: func_data[1])
def test_loss_func(score_list, func_data):
    """ Test score function """
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(score_list, dims=(0, 1))
    res_mt = func(masked_tensor)
    res_lst = func(torch.stack(score_list))
    assert torch.allclose(res_mt, res_lst, atol=ATOL), torch.norm(res_mt - res_lst, p=float('inf'))

TEST_ACCURACY_FUNCS = [
    #(accuracy_wrapper(accuracy_linear_assignment), 'accuracy_linear_assignment'),
    (accuracy_wrapper(accuracy_max), 'accuracy_max')]

@pytest.mark.parametrize('func_data', TEST_ACCURACY_FUNCS, ids=lambda func_data: func_data[1])
def test_accuracy_func(tensor_list, func_data):
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(1, 2))
    res_mt = func(masked_tensor)
    res_lst = sum(apply_list_tensors(tensor_list, func))
    assert torch.allclose(res_mt, res_lst, atol=ATOL), torch.norm(res_mt - res_lst, p=float('inf'))
