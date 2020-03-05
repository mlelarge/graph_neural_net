"""
Tests for maskedtensor module
To execute, run pytest
"""

import pytest
import torch
import torch.nn as nn
import maskedtensor
from models.base_model import Simple_Node_Embedding

def apply_list_tensors(lst, func):
    """ Apply func on each tensor (with batch dim) """
    batched_lst = [tens.unsqueeze(0) for tens in lst]
    batched_res_lst = [func(tens) for tens in batched_lst]
    res_lst = [tens.squeeze(0) for tens in batched_res_lst]
    return res_lst

N_FEATURES = 16
N_VERTICES_RANGE = range(10, 50)

@pytest.fixture
def tensor_list():
    """ Generate list of tensors (with graph convention) """
    lst = [torch.empty((n_vertices, n_vertices, N_FEATURES)).normal_()
           for n_vertices in N_VERTICES_RANGE]
    return lst

class GraphConv2d(nn.Module):
    """ Conv2d module with our graph convention for testing """
    def __init__(self, in_features, out_features):
        super(GraphConv2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x

TEST_FUNCS = [
    lambda t: torch.add(t, 1),
    lambda t: torch.mul(t, 2),
    lambda t: torch.sum(t, 2),
    lambda t: torch.max(t, 2)[0],
    # keep first dim not to perturb apply_list_tensors
    lambda t: t.permute(0, 3, 2, 1),
    GraphConv2d(N_FEATURES, 2*N_FEATURES)]

@pytest.mark.parametrize('func', TEST_FUNCS)
def test_torch_func(tensor_list, func):
    """ Test torch function """
    masked_tensor = maskedtensor.from_list(tensor_list, dims=(0, 1))
    res_mt = list(func(masked_tensor))
    res_lst = apply_list_tensors(tensor_list, func)
    for t_mt, t_lst in zip(res_mt, res_lst):
        assert t_mt.size() == t_lst.size()
        assert torch.equal(t_mt, t_lst)
