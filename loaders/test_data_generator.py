import pytest
import networkx
import torch
import data_generator

N_VERTICES = 50
NOISE = 0.05
EDGE_DENSITY = 0.2

@pytest.fixture
def regular_graph():
    g, W = data_generator.generate_regular_graph_netx(EDGE_DENSITY, N_VERTICES)
    return g, W

def test_edge_swap_on_regular(regular_graph):
    g, W = regular_graph
    W_noise = data_generator.noise_edge_swap(g, W, NOISE, EDGE_DENSITY)
    degrees = torch.sum(W, 0)
    degrees_noise = torch.sum(W_noise, 0)
    assert torch.equal(degrees, degrees_noise)
