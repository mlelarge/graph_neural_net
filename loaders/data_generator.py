import numpy as np
import os
import networkx
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from toolbox import utils

class Generator(torch.utils.data.Dataset):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args):
        self.name = name
        if name == 'train':
            self.num_examples = args['--num_examples_train']
        elif name == 'test':
            self.num_examples = args['--num_examples_test']
        elif name == 'val':
            self.num_examples = args['--num_examples_val']
        self.data = []
        self.n_vertices = args['--n_vertices']
        self.generative_model = args['--generative_model']
        self.edge_density = args['--edge_density']
        self.noise = args['--noise']
        subfolder_name = 'QAP_{}_{}_{}_{}'.format(self.generative_model,
                                                  self.num_examples,
                                                  self.n_vertices,
                                                  self.noise, self.edge_density)
        self.path_dataset = os.path.join(args['--path_dataset'],
                                         subfolder_name)
        utils.check_dir(self.path_dataset)

    def ErdosRenyi_netx(self, p, N):
        g = networkx.erdos_renyi_graph(N, p)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def BarabasiAlbert_netx(self, p, N):
        m = int(p*(N -1)/2)
        g = networkx.barabasi_albert_graph(N, m)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def RegularGraph_netx(self, p, N):
        """ Generate random regular graph """
        d = p * N
        d = int(d)
        g = networkx.random_regular_graph(d, N)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def adjacency_matrix_to_tensor_representation(self, W):
        """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
        degrees = W.sum(1)
        B = np.zeros((self.n_vertices, self.n_vertices, 2))
        B[:, :, 1] = W
        indices = np.arange(self.n_vertices)
        B[indices, indices, 0] = degrees
        return B

    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi_netx(self.edge_density, self.n_vertices)
        elif self.generative_model == 'Regular':
            W = self.RegularGraph_netx(self.edge_density, self.n_vertices)
        elif self.generative_model == 'BarabasiAlbert':
            W = self.BarabasiAlbert_netx(self.edge_density, self.n_vertices)

        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        pe1 = self.noise
        pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
        noise1 = self.ErdosRenyi_netx(pe1, self.n_vertices)
        noise2 = self.ErdosRenyi_netx(pe2, self.n_vertices)
        W_noise = W*(1-noise1) + (1-W)*noise2
        B = self.adjacency_matrix_to_tensor_representation(W)
        B_noise = self.adjacency_matrix_to_tensor_representation(W_noise)
        return (B, B_noise)

    def create_dataset(self):
        for i in range(self.num_examples):
            example = self.compute_example()
            self.data.append(np.asarray(example))

    def load_dataset(self, saving=True):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            with open(path, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                self.data = list(data)
        else:
            print('Creating dataset.')
            self.data = []
            self.create_dataset()
            print('Saving datatset at {}'.format(path))
            with open(path, 'wb') as f:
                    np.save(f, self.data)
        return self.data

    def __getitem__(self ,i):

        return (torch.Tensor(self.data[i][0]), torch.Tensor(self.data[i][1]))
    
    def __len__(self):
        return len(self.data)

