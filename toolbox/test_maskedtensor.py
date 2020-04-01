"""
Tests for maskedtensor module
To execute, run pyton -m pytest at the root of the project

Recommanded: install pytest-repeat to repeat tests with e.g.
    python -m pytest . --count 10
"""

import functools
import pytest
import torch
import torch.nn as nn
import maskedtensor
from models.layers import MlpBlock, RegularBlock
from models.base_model import Simple_Node_Embedding
from models.siamese_net import Siamese_Model
from losses import get_criterion
from metrics import accuracy_linear_assignment, accuracy_max

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
    """ Generate list of tensors (with graph convention) """
    lst = [torch.empty((n_vertices, n_vertices, N_FEATURES)).normal_()
           for n_vertices in N_VERTICES_RANGE]
    return lst

@pytest.fixture
def score_list():
    """ Generate list of tensors with no features and fixed n_vertices"""
    lst = [torch.empty((FIXED_N_VERTICES, FIXED_N_VERTICES)).normal_()
           for _ in N_VERTICES_RANGE]
    return lst

other_tensor_list = tensor_list

def graph_conv_wrapper(func):
    """ Applies graph convention to a function using pytorch convention """
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        new_args = [x.permute(0, 3, 1, 2) for x in args]
        ret = func(*new_args, **kwargs)
        return ret.permute(0, 2, 3, 1)
    return wrapped_func

def accuracy_wrapper(func):
    """ Wraps accuracy funcs so that they behave like the other funcs """
    @functools.wraps(func)
    def wrapped_func(weights, *args, **kwargs):
        # remove features
        new_weights = torch.sum(weights, -1)
        ret = func(new_weights, *args, **kwargs)
        return torch.Tensor([ret[0]])
    return wrapped_func

# the third parameter specifies whether the base_name of the second maskedtensor
# should match the first one
TEST_BINARY_FUNCS = [
        # when pytorch issue is fixed, change this
        (lambda t1, t2: maskedtensor.dispatch_cat((t1, t2), dim=-1), 'torch.cat', True),
        (graph_conv_wrapper(torch.matmul), 'torch.matmul', True),
        (graph_conv_wrapper(torch.matmul), 'torch.matmul', False),
        (Siamese_Model(N_FEATURES, 2, 32, 32, 3), 'Siamese_Model', False)]

@pytest.mark.parametrize('func_data', TEST_BINARY_FUNCS, ids=lambda func_data: func_data[1])
def test_binary_torch_func(tensor_list, other_tensor_list, func_data):
    """ Test torch function wich use two tensors """
    func, _, same_base_name  = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(0, 1))
    other_base_name = 'N' if same_base_name else 'M'
    other_masked_tensor = maskedtensor.from_list(other_tensor_list, dims=(0, 1),
            base_name=other_base_name)
    res_mt = list(func(masked_tensor, other_masked_tensor))
    binary_list = zip(tensor_list, other_tensor_list)
    res_lst = apply_binary_list_tensors(binary_list, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.allclose(t_mt, t_lst, atol=ATOL), torch.norm(t_mt - t_lst, p=float('inf'))

TEST_FUNCS = [
    (lambda t: torch.add(t, 1), 'torch.add'),
    (lambda t: torch.mul(t, 2), 'torch.mul'),
    (lambda t: torch.sum(t, 2), 'torch.sum'),
    (lambda t: torch.max(t, 2)[0], 'torch.max'),
    # keep first dim not to perturb apply_list_tensors
    (lambda t: t.permute(0, 3, 2, 1), 'permute'),
    (graph_conv_wrapper(nn.Conv2d(N_FEATURES, 2*N_FEATURES, 1)), 'nn.Conv2d'),
    (graph_conv_wrapper(MlpBlock(N_FEATURES, 2*N_FEATURES, 2)), 'MlpBlock'),
    (graph_conv_wrapper(RegularBlock(N_FEATURES, 2*N_FEATURES, 2)), 'RegularBlock'),
    (Simple_Node_Embedding(N_FEATURES, 2, 32, 32, 3), 'Simple_Node_Embedding')]

@pytest.mark.parametrize('func_data', TEST_FUNCS, ids=lambda func_data: func_data[1])
def test_torch_func(tensor_list, func_data):
    """ Test torch function """
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(0, 1))
    res_mt = list(func(masked_tensor))
    res_lst = apply_list_tensors(tensor_list, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.allclose(t_mt, t_lst, atol=ATOL), torch.norm(t_mt - t_lst, p=float('inf'))

TEST_SCORE_FUNCS = [
        (get_criterion(DEVICE, 'mean'), 'loss')]

@pytest.mark.parametrize('func_data', TEST_SCORE_FUNCS, ids=lambda func_data: func_data[1])
def test_score_func(score_list, func_data):
    """ Test score function """
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(score_list, dims=(0, 1))
    res_mt = func(masked_tensor)
    res_lst = func(torch.stack(score_list))
    assert torch.allclose(res_mt, res_lst, atol=ATOL), torch.norm(res_mt - res_lst, p=float('inf'))

TEST_ACCURACY_FUNCS = [
    (accuracy_wrapper(accuracy_linear_assignment), 'accuracy_linear_assignment'),
    (accuracy_wrapper(accuracy_max), 'accuracy_max')]

@pytest.mark.parametrize('func_data', TEST_ACCURACY_FUNCS, ids=lambda func_data: func_data[1])
def test_accuracy_func(tensor_list, func_data):
    func, _ = func_data
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(0, 1))
    res_mt = func(masked_tensor)
    res_lst = sum(apply_list_tensors(tensor_list, func))
    assert torch.allclose(res_mt, res_lst, atol=ATOL), torch.norm(res_mt - res_lst, p=float('inf'))
