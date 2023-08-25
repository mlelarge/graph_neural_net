import pytest
import torch
from toolbox.metrics import accuracy_linear_assignment, accuracy_max
from maskedtensor import from_list

N_VERTICES_RANGE = range(40, 50)

@pytest.fixture
def correct_batch():
    tensor_lst = [torch.eye(n_vertices) for n_vertices in N_VERTICES_RANGE]
    return from_list(tensor_lst, dims=(0, 1))

TEST_ACCURACY_FUNCS = [
    #(accuracy_linear_assignment, 'accuracy_linear_assignment'),
    (accuracy_max, 'accuracy_max')]

@pytest.mark.parametrize('func_data', TEST_ACCURACY_FUNCS, ids=lambda func_data: func_data[1])
def test_perfect_accuracy(correct_batch, func_data):
    func, _ = func_data
    correct, total = func(correct_batch)
    assert correct == total, (correct, total)

@pytest.fixture
def batch():
    tensor_lst = [torch.empty(n_vertices, n_vertices).normal_() for n_vertices in N_VERTICES_RANGE]
    return from_list(tensor_lst, dims=(0, 1))
