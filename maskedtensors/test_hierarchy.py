import pytest
import torch
from toolbox.metrics import accuracy_max

from toolbox.losses import triplet_loss
import maskedtensor
import math
import scipy.optimize

N_VERTICES_RANGE = range(40, 50)
DEVICE = torch.device('cpu')
OPT_SCALE = True

def perturb(target):
    target[0, :] = 2
    return target

@pytest.fixture
def batch(request):
    transpose = request.param
    if transpose:
        tensor_lst = [torch.t(perturb(torch.eye(n_vertices, n_vertices)))
                      for n_vertices in N_VERTICES_RANGE]
    else:
        tensor_lst = [perturb(torch.eye(n_vertices, n_vertices)) for n_vertices in N_VERTICES_RANGE]
    return maskedtensor.from_list(tensor_lst, dims=(0, 1))

@pytest.mark.parametrize('batch', [False, True], indirect=['batch'])
def test_hierarchy(batch):
    correct, total = accuracy_max(batch)
    acc = correct/total
    loss_func = triplet_loss(loss_reduction='mean')
    if OPT_SCALE:
        res = scipy.optimize.minimize_scalar(lambda x: loss_func(torch.mul(batch, x)), bracket=(1e-1, 1e2))
        scale = res.x
        if scale <= 0:
            raise RuntimeError("Something went wrong during the optimization process")
    else:
        scale = 216
    loss = loss_func(torch.mul(batch, scale))
    assert loss >= (1 - acc) * math.log(2)
