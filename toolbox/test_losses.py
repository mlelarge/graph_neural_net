import pytest
import torch
import losses
import maskedtensor

BATCH_SIZE = 32
N_VERTICES = 50

@pytest.fixture
def std_batch():
    tensor = torch.empty((BATCH_SIZE, N_VERTICES, N_VERTICES))
    tensor = tensor.normal_()
    return tensor

@pytest.fixture
def masked_batch():
    lst = [torch.empty((N_VERTICES, N_VERTICES)).normal_()
           for _ in range(BATCH_SIZE)]
    mtensor = maskedtensor.from_list(lst, dims=(0, 1))
    return mtensor

@pytest.fixture
def batch(request):
    return request.getfixturevalue(request.param)

@pytest.mark.parametrize('batch', ['std_batch', 'masked_batch'], indirect=True)
def test_loss_fixed_size(batch):
    device = torch.device('cpu')
    loss_func_mean = losses.get_criterion(device, 'mean')
    loss_func_mean_of_mean = losses.get_criterion(device, 'mean_of_mean')
    loss_mean = loss_func_mean(batch)
    loss_mean_of_mean = loss_func_mean_of_mean(batch)
    assert loss_mean.size() == loss_mean_of_mean.size()
    assert torch.allclose(loss_mean, loss_mean_of_mean), loss_mean - loss_mean_of_mean
