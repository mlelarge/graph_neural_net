import os
import networkx
import torch
import torch.utils
from toolbox import utils

def generate_erdos_renyi_netx(p, N):
    """ Generate random Erdos Renyi graph """
    g = networkx.erdos_renyi_graph(N, p)
    W = networkx.adjacency_matrix(g).todense()
    return torch.as_tensor(W, dtype=torch.float)

def generate_barabasi_albert_netx(p, N):
    """ Generate random Barabasi Albert graph """
    m = int(p*(N -1)/2)
    g = networkx.barabasi_albert_graph(N, m)
    W = networkx.adjacency_matrix(g).todense()
    return torch.as_tensor(W, dtype=torch.float)

def generate_regular_graph_netx(p, N):
    """ Generate random regular graph """
    d = p * N
    d = int(d)
    g = networkx.random_regular_graph(d, N)
    W = networkx.adjacency_matrix(g).todense()
    return torch.as_tensor(W, dtype=torch.float)

def adjacency_matrix_to_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    degrees = W.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B

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


    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        if self.generative_model == 'ErdosRenyi':
            W = generate_erdos_renyi_netx(self.edge_density, self.n_vertices)
        elif self.generative_model == 'Regular':
            W = generate_regular_graph_netx(self.edge_density, self.n_vertices)
        elif self.generative_model == 'BarabasiAlbert':
            W = generate_barabasi_albert_netx(self.edge_density, self.n_vertices)

        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        pe1 = self.noise
        pe2 = (self.edge_density*self.noise)/(1-self.edge_density)
        noise1 = generate_erdos_renyi_netx(pe1, self.n_vertices)
        noise2 = generate_erdos_renyi_netx(pe2, self.n_vertices)
        W_noise = W*(1-noise1) + (1-W)*noise2
        B = adjacency_matrix_to_tensor_representation(W)
        B_noise = adjacency_matrix_to_tensor_representation(W_noise)
        return (B, B_noise)

    def create_dataset(self):
        for _ in range(self.num_examples):
            example = self.compute_example()
            self.data.append(example)

    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            data = torch.load(path)
            self.data = list(data)
        else:
            print('Creating dataset.')
            self.data = []
            self.create_dataset()
            print('Saving datatset at {}'.format(path))
            torch.save(self.data, path)

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)
