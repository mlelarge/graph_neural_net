"""
This code is 
adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch
"""

import numpy as np
import os
import pickle
from pathlib import Path
ROOT_DIR = Path.home()
DATA_DIR = os.path.join(ROOT_DIR,'data/')

NUM_LABELS = {'ENZYMES': 3, 'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38,
              'PROTEINS': 3, 'PTC': 22, 'DD': 89}
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_CLASSES = {'COLLAB':3, 'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2, 'PROTEINS':2, 'PTC':2, 'QM9': 12}


def load_dataset(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two lists of lenght (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex_labels+1, num_vertex, num_vertex)
            the labels array in index i represent the class of graphs[i]
    """
    directory = DATA_DIR + "benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(NUM_LABELS[ds_name]+2, num_vertex, num_vertex), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[int(vertex[0])+1, j, j]= 1.
                for k in range(2,len(vertex)):
                    curr_graph[0, j, int(vertex[k])] = 1.
            #print(curr_graph.shape)
            #curr_graph = normalize_graph(curr_graph)
            graphs.append(curr_graph)
    #graphs = np.array(graphs)
    #for i in range(graphs.shape[0]):
    #    graphs[i] = np.transpose(graphs[i], [2,0,1])
    return graphs, labels#np.array(labels)


def load_qm9(target_param):
    """
    Constructs the graphs and labels of QM9 data set, already split to train, val and test sets
    :return: 6 numpy arrays:
                 train_graphs: N_train,
                 train_labels: N_train x 12, (or Nx1 is target_param is not False)
                 val_graphs: N_val,
                 val_labels: N_train x 12, (or Nx1 is target_param is not False)
                 test_graphs: N_test,
                 test_labels: N_test x 12, (or Nx1 is target_param is not False)
                 each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    train_graphs, train_labels = load_qm9_aux('train', target_param)
    val_graphs, val_labels = load_qm9_aux('val', target_param)
    test_graphs, test_labels = load_qm9_aux('test', target_param)
    return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels


def load_qm9_aux(which_set, target_param):
    """
    Read and construct the graphs and labels of QM9 data set, already split to train, val and test sets
    :param which_set: 'test', 'train' or 'val'
    :param target_param: if not false, return the labels for this specific param only
    :return: graphs: (N,)
             labels: N x 12, (or Nx1 is target_param is not False)
             each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    base_path = BASE_DIR + "/data/QM9/QM9_{}.p".format(which_set)
    graphs, labels = [], []
    with open(base_path, 'rb') as f:
        data = pickle.load(f)
        for instance in data:
            labels.append(instance['y'])
            nodes_num = instance['usable_features']['x'].shape[0]
            graph = np.empty((nodes_num, nodes_num, 19))
            for i in range(13):
                # 13 features per node - for each, create a diag matrix of it as a feature
                graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
            graph[:, :, 13] = instance['usable_features']['distance_mat']
            graph[:, :, 14] = instance['usable_features']['affinity']
            graph[:, :, 15:] = instance['usable_features']['edge_features']  # shape n x n x 4
            graphs.append(graph)
    graphs = np.array(graphs)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])
    labels = np.array(labels).squeeze()  # shape N x 12
    if target_param is not False:  # regression over a specific target, not all 12 elements
        labels = labels[:, target_param].reshape(-1, 1)  # shape N x 1

    return graphs, labels


def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = DATA_DIR + "benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = DATA_DIR + "benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


# def group_same_size(graphs, labels):
#     """
#     group graphs of same size to same array
#     :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
#     :param labels: numpy array of labels
#     :return: two numpy arrays. graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
#             in the shape (number of graphs with this size, num vertex, num. vertex, num vertex labels)
#             the second arrayy is labels with correspons shape
#     """
#     sizes = list(map(lambda t: t.shape[1], graphs))
#     indexes = np.argsort(sizes)
#     graphs = graphs[indexes]
#     labels = labels[indexes]
#     r_graphs = []
#     r_labels = []
#     one_size = []
#     start = 0
#     size = graphs[0].shape[1]
#     for i in range(len(graphs)):
#         if graphs[i].shape[1] == size:
#             one_size.append(np.expand_dims(graphs[i], axis=0))
#         else:
#             r_graphs.append(np.concatenate(one_size, axis=0))
#             r_labels.append(np.array(labels[start:i]))
#             start = i
#             one_size = []
#             size = graphs[i].shape[1]
#             one_size.append(np.expand_dims(graphs[i], axis=0))
#     r_graphs.append(np.concatenate(one_size, axis=0))
#     r_labels.append(np.array(labels[start:]))
#     return r_graphs, r_labels


# helper method to shuffle each same size graphs array
# def shuffle_same_size(graphs, labels):
#     r_graphs, r_labels = [], []
#     for i in range(len(labels)):
#         curr_graph, curr_labels = shuffle(graphs[i], labels[i])
#         r_graphs.append(curr_graph)
#         r_labels.append(curr_labels)
#     return r_graphs, r_labels


# def split_to_batches(graphs, labels, size):
#     """
#     split the same size graphs array to batches of specified size
#     last batch is in size num_of_graphs_this_size % size
#     :param graphs: array of arrays of same size graphs
#     :param labels: the corresponding labels of the graphs
#     :param size: batch size
#     :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
#                 corresponds labels
#     """
#     r_graphs = []
#     r_labels = []
#     for k in range(len(graphs)):
#         r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
#         r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])

#     # Avoid bug for batch_size=1, where instead of creating numpy array of objects, we had numpy array of floats with
#     # different sizes - could not reshape
#     ret1, ret2 = np.empty(len(r_graphs), dtype=object), np.empty(len(r_labels), dtype=object)
#     ret1[:] = r_graphs
#     ret2[:] = r_labels
#     return ret1, ret2


# helper method to shuffle the same way graphs and labels arrays
# def shuffle(graphs, labels):
#     shf = np.arange(labels.shape[0], dtype=np.int32)
#     np.random.shuffle(shf)
#     return np.array(graphs)[shf], labels[shf]


# def normalize_graph(curr_graph):

#     split = np.split(curr_graph, [1], axis=2)

#     adj = np.squeeze(split[0], axis=2)
#     deg = np.sqrt(np.sum(adj, 0))
#     deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
#     normal = np.diag(deg)
#     norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
#     ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
#     spred_adj = np.multiply(ones, norm_adj)
#     labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
#     return np.add(spred_adj, labels)


# if __name__ == '__main__':
#     graphs, labels = load_dataset("MUTAG")
#     a, b = get_train_val_indexes(1, "MUTAG")
#     print(np.transpose(graphs[a[0]], [1, 2, 0])[0])
