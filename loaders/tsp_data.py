import time
import pickle
import numpy as np
import itertools
from scipy.spatial.distance import pdist, squareform

import torch
from torch.utils.data import Dataset


def adjacency_matrix_to_tensor_representation(W, mask, with_degree= True):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    degrees = W.sum(1)
    n_nodes ,_  = W.shape
    rescale = torch.sqrt(torch.from_numpy(np.array([n_nodes], dtype=np.float)))#.mean(W)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W*rescale
    indices = torch.arange(len(W))
    if with_degree:
        B[indices, indices, 0] = degrees
    return B


class TSP(Dataset):
    def __init__(self, data_dir, file_name='tsp_50_', split="train", num_neighbors=49, max_samples=10000):    
        self.data_dir = data_dir
        self.split = split
        self.filename = f'{data_dir}/{file_name}{split}.txt'
        self.max_samples = max_samples
        self.num_neighbors = num_neighbors
        self.is_test = split.lower() in ['test', 'val']
        
        self.graph_lists = []
        self.edge_labels = []
        self.mask_lists = []
        self._prepare()
        self.n_samples = len(self.edge_labels)
    
    def _prepare(self):
        print('preparing all graphs for the %s set...' % self.split.upper())
        
        file_data = open(self.filename, "r").readlines()[:self.max_samples]
        
        for line in file_data:
            line = line.split(" ")  # Split into list
            num_nodes = int(line.index('output')//2)
            
            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * num_nodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])

            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            # Determine k-nearest neighbors for each node
            knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
            W_tensor = torch.as_tensor(W_val, dtype=torch.float)
            knns = torch.as_tensor(knns.copy())
            mask = torch.zeros((num_nodes,num_nodes))
            mask.scatter_(1,torch.as_tensor(knns),1)
            

            ## Convert tour nodes to required format
            #tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

            #edge_labels = -np.ones((num_nodes,2))
            #for idx in range(len(tour_nodes)-1):
            #    edge_labels[tour_nodes[idx],0] = tour_nodes[idx+1]
            #    edge_labels[tour_nodes[idx],1] = tour_nodes[idx-1]
                
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]

            # Compute an edge adjacency matrix representation of tour
            edges_target = np.eye(num_nodes)
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                edges_target[i][j] = 1
                edges_target[j][i] = 1
            ## Add final connection of tour in edge target
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1

            self.graph_lists.append(adjacency_matrix_to_tensor_representation(W_tensor,mask,False))
            self.mask_lists.append(mask)
            self.edge_labels.append(torch.as_tensor(edges_target))

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
        """
        return self.graph_lists[idx], self.edge_labels[idx], self.mask_lists[idx]