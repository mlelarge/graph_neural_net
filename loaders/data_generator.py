import os
import random
import itertools
import networkx
from networkx.algorithms.approximation.clique import max_clique
import torch
import torch.utils
import toolbox.utils as utils
from concorde.tsp import TSPSolver
import math
from toolbox.searches import mcp_beam_method
import timeit
from sklearn.decomposition import PCA
from numpy import pi,angle,cos,sin
import tqdm

GENERATOR_FUNCTIONS = {}
GENERATOR_FUNCTIONS_TSP = {}

def generates(name):
    """ Register a generator function for a graph distribution """
    def decorator(func):
        GENERATOR_FUNCTIONS[name] = func
        return func
    return decorator

@generates("ErdosRenyi")
def generate_erdos_renyi_netx(p, N):
    """ Generate random Erdos Renyi graph """
    g = networkx.erdos_renyi_graph(N, p)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("BarabasiAlbert")
def generate_barabasi_albert_netx(p, N):
    """ Generate random Barabasi Albert graph """
    m = int(p*(N -1)/2)
    g = networkx.barabasi_albert_graph(N, m)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("Regular")
def generate_regular_graph_netx(p, N):
    """ Generate random regular graph """
    d = p * N
    d = int(d)
    # Make sure N * d is even
    if N * d % 2 == 1:
        d += 1
    g = networkx.random_regular_graph(d, N)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

NOISE_FUNCTIONS = {}

def noise(name):
    """ Register a noise function """
    def decorator(func):
        NOISE_FUNCTIONS[name] = func
        return func
    return decorator

@noise("ErdosRenyi")
def noise_erdos_renyi(g, W, noise, edge_density):
    n_vertices = len(W)
    pe1 = noise
    pe2 = (edge_density*noise)/(1-edge_density)
    _,noise1 = generate_erdos_renyi_netx(pe1, n_vertices)
    _,noise2 = generate_erdos_renyi_netx(pe2, n_vertices)
    W_noise = W*(1-noise1) + (1-W)*noise2
    return W_noise

def is_swappable(g, u, v, s, t):
    """
    Check whether we can swap
    the edges u,v and s,t
    to get u,t and s,v
    """
    actual_edges = g.has_edge(u, v) and g.has_edge(s, t)
    no_self_loop = (u != t) and (s != v)
    no_parallel_edge = not (g.has_edge(u, t) or g.has_edge(s, v))
    return actual_edges and no_self_loop and no_parallel_edge

def do_swap(g, u, v, s, t):
    g.remove_edge(u, v)
    g.remove_edge(s, t)
    g.add_edge(u, t)
    g.add_edge(s, v)

@noise("EdgeSwap")
def noise_edge_swap(g, W, noise, edge_density): #Permet de garder la regularite
    g_noise = g.copy()
    edges_iter = list(itertools.chain(iter(g.edges), ((v, u) for (u, v) in g.edges)))
    for u,v in edges_iter:
        if random.random() < noise:
            for s, t in edges_iter:
                if random.random() < noise and is_swappable(g_noise, u, v, s, t):
                    do_swap(g_noise, u, v, s, t)
    W_noise = networkx.adjacency_matrix(g_noise).todense()
    return torch.as_tensor(W_noise, dtype=torch.float)

def adjacency_matrix_to_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    degrees = W.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B

##TSP Generation functions

def dist_from_pos(pos):
    N = len(pos)
    W_dist = torch.zeros((N,N))
    for i in range(0,N-1):
        for j in range(i+1,N):
            curr_dist = math.sqrt( (pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
            W_dist[i,j] = curr_dist
            W_dist[j,i] = curr_dist
    return W_dist

def generates_TSP(name):
    """ Register a generator function for a graph distribution """
    def decorator(func):
        GENERATOR_FUNCTIONS_TSP[name] = func
        return func
    return decorator

@generates_TSP("GaussNormal")
def generate_gauss_normal_netx(N):
    """ Generate random graph with points"""
    pos = {i: (random.gauss(0, 1), random.gauss(0, 1)) for i in range(N)} #Define the positions of the points
    W_dist = dist_from_pos(pos)
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g.add_edges_from(networkx.complete_graph(N).edges)
    return g, torch.as_tensor(W_dist, dtype=torch.float)

@generates_TSP("Square01")
def generate_square_netx(N):
    pos = {i: (random.random(), random.random()) for i in range(N)} #Define the positions of the points
    W_dist = dist_from_pos(pos)
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g.add_edges_from(networkx.complete_graph(N).edges)
    return g, torch.as_tensor(W_dist, dtype=torch.float)

def distance_matrix_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    W_adjacency = torch.sign(W)
    degrees = W_adjacency.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B

def normalize_tsp(xs,ys):
    """ 'Normalizes' points positions by moving they in a way where the principal component of the point cloud is directed vertically"""
    X = [(x,y) for x,y in zip(xs,ys)]
    pca = PCA(n_components=1)
    pca.fit(X)
    pc = pca.components_[0]
    rot_angle = pi/2 - angle(pc[0]+1j*pc[1])
    x_rot = [ x*cos(rot_angle) - y*sin(rot_angle) for x,y in X ]
    y_rot = [ x*sin(rot_angle) + y*cos(rot_angle) for x,y in X ]
    return x_rot,y_rot

class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples

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
            print('Saving dataset at {}'.format(path))
            torch.save(self.data, path)
    
    def create_dataset(self):
        for _ in tqdm.tqdm(range(self.num_examples)):
            example = self.compute_example()
            self.data.append(example)

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)

class QAP_Generator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args):
        self.generative_model = args['generative_model']
        self.noise_model = args['noise_model']
        self.edge_density = args['edge_density']
        self.noise = args['noise']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        subfolder_name = 'QAP_{}_{}_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     self.noise_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba,
                                                     self.noise, self.edge_density)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        
        
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        try:
            g, W = GENERATOR_FUNCTIONS[self.generative_model](self.edge_density, n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        try:
            W_noise = NOISE_FUNCTIONS[self.noise_model](g, W, self.noise, self.edge_density)
        except KeyError:
            raise ValueError('Noise model {} not supported'
                             .format(self.noise_model))
        B = adjacency_matrix_to_tensor_representation(W)
        B_noise = adjacency_matrix_to_tensor_representation(W_noise)
        data = torch.cat((B.unsqueeze(0),B_noise.unsqueeze(0)))
        return (data,torch.empty(0)) #Empty tensor used as dummy data

class TSP_Generator(Base_Generator):
    """
    Traveling Salesman Problem Generator.
    Uses the pyconcorde wrapper : see https://github.com/jvkersch/pyconcorde (thanks a lot)
    """
    def __init__(self, name, args, coeff=1e8):
        self.generative_model = args['generative_model']
        self.distance = args['distance_used']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_{}_{}_{}_{}'.format(self.generative_model, 
                                                     self.distance,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = True
        self.coeff = coeff
        self.positions = []
    
    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            data,pos = torch.load(path)
            self.data = list(data)
            self.positions = list(pos)
        else:
            print('Creating dataset.')
            self.data = []
            self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save((self.data,self.positions), path)

    def compute_example(self):
        """
        Compute pairs (Adjacency, Optimal Tour)
        """
        try:
            g, W = GENERATOR_FUNCTIONS_TSP[self.generative_model](self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        xs = [g.nodes[node]['pos'][0] for node in g.nodes]
        ys = [g.nodes[node]['pos'][1] for node in g.nodes]

        problem = TSPSolver.from_data([self.coeff*elt for elt in xs],[self.coeff*elt for elt in ys],self.distance) #1e8 because Concorde truncates the distance to the nearest integer
        solution = problem.solve(verbose=False)
        assert solution.success, f"Couldn't find solution! \n x =  {xs} \n y = {ys} \n {solution}"

        B = distance_matrix_tensor_representation(W)
        
        SOL = torch.zeros((self.n_vertices,self.n_vertices),dtype=torch.float)
        prec = solution.tour[-1]
        for i in range(self.n_vertices):
            curr = solution.tour[i]
            SOL[curr,prec] = 1
            SOL[prec,curr] = 1
            prec = curr
        
        self.positions.append((xs,ys))
        return (B, SOL)

class TSP_normalized_Generator(Base_Generator):
    """
    Traveling Salesman Problem Generator.
    Uses the pyconcorde wrapper : see https://github.com/jvkersch/pyconcorde (thanks a lot)
    """
    def __init__(self, name, args, coeff=1e8):
        self.generative_model = args['generative_model']
        self.distance = args['distance_used']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_normed_{}_{}_{}_{}'.format(self.generative_model, 
                                                     self.distance,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = True
        self.coeff = coeff
        self.positions = []
    
    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            data,pos = torch.load(path)
            self.data = list(data)
            self.positions = list(pos)
        else:
            print('Creating dataset.')
            self.data = []
            self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save((self.data,self.positions), path)

    def compute_example(self):
        """
        Compute pairs (Adjacency, Optimal Tour)
        """
        try:
            g, W = GENERATOR_FUNCTIONS_TSP[self.generative_model](self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        xs = [g.nodes[node]['pos'][0] for node in g.nodes]
        ys = [g.nodes[node]['pos'][1] for node in g.nodes]
        xs,ys = normalize_tsp(xs,ys)
        problem = TSPSolver.from_data([self.coeff*elt for elt in xs],[self.coeff*elt for elt in ys],self.distance) #1e8 because Concorde truncates the distance to the nearest integer
        solution = problem.solve(verbose=False)
        assert solution.success, f"Couldn't find solution! \n x =  {xs} \n y = {ys} \n {solution}"

        B = distance_matrix_tensor_representation(W)
        
        SOL = torch.zeros((self.n_vertices,self.n_vertices),dtype=torch.float)
        prec = solution.tour[-1]
        for i in range(self.n_vertices):
            curr = solution.tour[i]
            SOL[curr,prec] = 1
            SOL[prec,curr] = 1
            prec = curr
        
        self.positions.append((xs,ys))
        return (B, SOL)

class TSP_custom_Generator(Base_Generator):
    """
    This class just has the 'custom_example' method. It is useful for testing a model on custom examples.
    """
    def __init__(self, name, args, coeff=1e8):
        self.distance = args['distance_used']
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_custom_{}_{}'.format(self.distance,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples=1)
        self.data = []
        
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = True
        self.coeff = coeff
        self.positions = []
    
    def custom_example(self,xs,ys):
        n = len(xs)
        pos = {i: (xs[i], ys[i]) for i in range(n)} #Define the positions of the points
        W_dist = dist_from_pos(pos)
        g = networkx.random_geometric_graph(n,0,pos=pos)
        g.add_edges_from(networkx.complete_graph(n).edges)
        W = torch.as_tensor(W_dist, dtype=torch.float)

        problem = TSPSolver.from_data([self.coeff*elt for elt in xs],[self.coeff*elt for elt in ys],self.distance) #1e8 because Concorde truncates the distance to the nearest integer
        solution = problem.solve(verbose=False)
        assert solution.success, f"Couldn't find solution! \n x =  {xs} \n y = {ys} \n {solution}"

        B = distance_matrix_tensor_representation(W)
        
        SOL = torch.zeros((n,n),dtype=torch.float)
        prec = solution.tour[-1]
        for i in range(n):
            curr = solution.tour[i]
            SOL[curr,prec] = 1
            SOL[prec,curr] = 1
            prec = curr
        
        self.positions.append((xs,ys))
        self.data.append((B,SOL))

class TSP_RL_Generator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args, coeff=1e8):
        self.generative_model = args['generative_model']
        self.distance = args['distance_used']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_RL_{}_{}_{}_{}'.format(self.generative_model, 
                                                     self.distance,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = True
        self.coeff = coeff
        self.positions = []

    def compute_example(self):
        """
        Compute pairs (Adjacency, Optimal Tour)
        """
        try:
            g, W = GENERATOR_FUNCTIONS_TSP[self.generative_model](self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        xs = [g.nodes[node]['pos'][0] for node in g.nodes]
        ys = [g.nodes[node]['pos'][1] for node in g.nodes]

        W += 1000*torch.eye(self.n_vertices) #~to adding infinity on the edges from a node to itself

        B = distance_matrix_tensor_representation(W)
        
        self.positions.append((xs,ys))
        return (B, W) # Keep the distance matrix W in the target for the loss to be computed

class TSP_Distance_Generator(Base_Generator):
    """
    Traveling Salesman Problem Generator.
    Uses the pyconcorde wrapper : see https://github.com/jvkersch/pyconcorde (thanks a lot)
    This version uses a vertex distance matrix as the target instead of an adjacency matrix
    """
    def __init__(self, name, args, coeff=1e8):
        self.generative_model = args['generative_model']
        self.distance = args['distance_used']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_DIST_{}_{}_{}_{}'.format(self.generative_model, 
                                                     self.distance,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = True
        self.coeff = coeff
        self.positions = []
    
    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            data,pos = torch.load(path)
            self.data = list(data)
            self.positions = list(pos)
        else:
            print('Creating dataset.')
            self.data = []
            self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save((self.data,self.positions), path)

    def compute_example(self):
        """
        Compute pairs (Adjacency, Optimal Tour)
        """
        try:
            g, W = GENERATOR_FUNCTIONS_TSP[self.generative_model](self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        xs = [g.nodes[node]['pos'][0] for node in g.nodes]
        ys = [g.nodes[node]['pos'][1] for node in g.nodes]

        problem = TSPSolver.from_data([self.coeff*elt for elt in xs],[self.coeff*elt for elt in ys],self.distance) #1e8 because Concorde truncates the distance to the nearest integer
        solution = problem.solve(verbose=False)
        assert solution.success, f"Couldn't find solution! \n x =  {xs} \n y = {ys} \n {solution}"

        tour = torch.Tensor(solution.tour)
        W_perm = utils.permute_adjacency(W,tour)

        B = distance_matrix_tensor_representation(W_perm)
        
        SOL = torch.zeros((self.n_vertices,self.n_vertices),dtype=torch.float)
        distance = torch.cat((torch.arange(self.n_vertices/2),torch.arange(self.n_vertices//2,0,-1))) #Will return a dtype=float, care
        distance /= self.n_vertices  #To have values between 0 and 1, for normalization
        for i in range(SOL.shape[0]): #For each vertex
            SOL[i,:] = distance.roll(i) #Create the distance matrix
        
        self.positions.append((xs,ys))
        return (B, SOL)

class MCP_Generator(Base_Generator):
    """
    Generator for the Maximum Clique Problem.
    This generator plants a clique of 'clique_size' size in the graph.
    It is then used as a seed to find a possible bigger clique with this seed
    """
    def __init__(self, name, args):
        self.edge_density = args['edge_density']
        self.clique_size = args['clique_size']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'MCP_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices, 
                                                           self.clique_size, 
                                                           self.edge_density)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        
        """
        try:
            g, W = GENERATOR_FUNCTIONS["ErdosRenyi"](self.edge_density, self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        W, K = self.add_clique(W,self.clique_size)
        B = adjacency_matrix_to_tensor_representation(W)


        k_size = len(torch.where(K.sum(dim=-1)!=0)[0])
        seed = utils.mcp_adj_to_ind(K)
        K2 = mcp_beam_method(B,K,seeds=seed,add_singles=False) #Finds a probably better solution with beam search in the form of a list of indices
        k2_size = len(K2)
        if k2_size>k_size:
            K = utils.mcp_ind_to_adj(K2,self.n_vertices)
        return (B, K)
        
    @classmethod
    def add_clique_base(cls,W,k):
        K = torch.zeros((len(W),len(W)))
        K[:k,:k] = torch.ones((k,k)) - torch.eye(k)
        W[:k,:k] = torch.ones((k,k)) - torch.eye(k)
        return W, K

    @classmethod
    def add_clique(cls,W,k):
        K = torch.zeros((len(W),len(W)))
        indices = random.sample(range(len(W)),k)
        l_indices = [(id_i,id_j) for id_i in indices for id_j in indices if id_i!=id_j] #Makes all the pairs of indices where we put the clique (get rid of diagonal terms)
        t_ind = torch.tensor(l_indices)
        K[t_ind[:,0],t_ind[:,1]] = 1
        W[t_ind[:,0],t_ind[:,1]] = 1
        return W,K

class MCP_True_Generator(Base_Generator):
    """
    Generator for the Maximum Clique Pb, which doesn't plant a clique
    """
    def __init__(self, name, args):
        self.edge_density = args['edge_density']
        self.clique_size = args['clique_size']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'MCP_true_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices, 
                                                           self.clique_size, 
                                                           self.edge_density)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        
        """
        try:
            g, W = GENERATOR_FUNCTIONS["ErdosRenyi"](self.edge_density, self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        
        mc = max_clique(g)
        l_indices = [(id_i,id_j) for id_i in mc for id_j in mc if id_i!=id_j]
        t_ind = torch.tensor(l_indices)
        K = torch.zeros_like(W)
        K[t_ind[:,0],t_ind[:,1]] = 1

        B = adjacency_matrix_to_tensor_representation(W)
        return (B, K)

class SBM_Generator(Base_Generator):
    def __init__(self, name, args):
        self.n_vertices = args['n_vertices']
        self.p_inter = args['p_inter']
        self.p_outer = args['p_outer']
        self.alpha = args['alpha']
        num_examples = args['num_examples_' + name]
        subfolder_name = 'SBM_{}_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices,
                                                           self.alpha, 
                                                           self.p_inter,
                                                           self.p_outer)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)
    
    def compute_example(self):
        """
        Computes the pair (data,target). Data is the adjacency matrix. For target, there are 2 interpretations :
         - if used with similarity : K_{ij} is 1 if node i and j are from the same group
         - if used with edge probas: K_{ij} is 1 if it's an intra-edge (so i and j from the same group)
        """
        n = self.n_vertices
        n_sub_a = n//2
        n_sub_b = n - n_sub_a # In case n_vertices is odd

        ga = (torch.empty((n_sub_a,n_sub_a)).uniform_()<(self.p_inter)).to(torch.float)
        gb = (torch.empty((n_sub_b,n_sub_b)).uniform_()<(self.p_inter)).to(torch.float)
        glink = (torch.empty((n_sub_a,n_sub_b)).uniform_()<self.p_outer).to(torch.float)
        
        adj = torch.zeros((self.n_vertices,self.n_vertices))

        adj[:n_sub_a,:n_sub_a] = ga.detach().clone()
        adj[:n_sub_a,n_sub_a:] = glink.detach().clone()
        adj[n_sub_a:,:n_sub_a] = glink.T.detach().clone()
        adj[n_sub_a:,n_sub_a:] = gb.detach().clone()

        K = torch.zeros((n,n))
        K[:n_sub_a,:n_sub_a] = 1
        K[n_sub_a:,n_sub_a:] = 1
        K,adj = utils.permute_adjacency_twin(K,adj)
        B = adjacency_matrix_to_tensor_representation(adj)
        return (B, K) 


if __name__=="__main__":
    #data_args = {"edge_density_a":0.3,"edge_density_b":0.2, "num_examples_train":5,"path_dataset":"dataset_sbm","n_vertices":10}
    #data_args = {"edge_density":0.7,"planted":True,'clique_size':11,"num_examples_train":1000,"path_dataset":"dataset_test","n_vertices":50}
    data_args = {"num_examples_train":10,"path_dataset":"dataset_test","n_vertices":13, 'distance_used':'EUC_2D','generative_model':'Square01'}
    
    g = TSP_Distance_Generator("train",data_args)
    timeit.timeit(g.load_dataset,number=1)

