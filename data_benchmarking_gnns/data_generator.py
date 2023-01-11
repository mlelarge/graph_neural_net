"""
This code is 
adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch
"""
import data_benchmarking_gnns.data_helper as helper
#import utils.config
import torch


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.hyperparams.batch_size
        self.is_qm9 = self.config.dataset_name == 'QM9'
        self.labels_dtype = torch.float32 if self.is_qm9 else torch.long

        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        if self.is_qm9:
            self.load_qm9_data()
        else:
            self.load_data_benchmark()

        self.split_val_test_to_batches()

    # load QM9 data set
    def load_qm9_data(self):
        train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels = \
            helper.load_qm9(self.config.target_param)

        # preprocess all labels by train set mean and std
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std

        self.train_graphs, self.train_labels = train_graphs, train_labels
        self.val_graphs, self.val_labels = val_graphs, val_labels
        self.test_graphs, self.test_labels = test_graphs, test_labels

        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        self.labels_std = train_labels_std  # Needed for postprocess, multiply mean abs distance by this std

    # load data for a benchmark graph (COLLAB, NCI1, NCI109, MUTAG, PTC, IMDBBINARY, IMDBMULTI, PROTEINS)
    def load_data_benchmark(self):
        graphs, labels = helper.load_dataset(self.config.dataset_name)
        # if no fold specify creates random split to train and validation
        if self.config.num_fold is None:
            graphs, labels = helper.shuffle(graphs, labels)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
                train_idx], graphs[test_idx], labels[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
                test_idx]
        # change validation graphs to the right shape
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        graphs, labels = next(self.iter)
        graphs, labels = torch.cuda.FloatTensor(graphs), torch.tensor(labels, device='cuda', dtype=self.labels_dtype)
        return graphs, labels

    # initialize an iterator from the data for one training epoch
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            self.iter = zip(self.val_graphs_batches, self.val_labels_batches)
        elif what_set == 'test':
            self.iter = zip(self.test_graphs_batches, self.test_labels_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs
        """
        graphs, labels = helper.group_same_size(self.train_graphs, self.train_labels)
        graphs, labels = helper.shuffle_same_size(graphs, labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels = helper.shuffle(graphs, labels)
        self.iter = zip(graphs, labels)

    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs, labels = helper.group_same_size(self.val_graphs, self.val_labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_val = len(graphs)
        self.val_graphs_batches, self.val_labels_batches = graphs, labels

        if self.is_qm9:
            # Benchmark graphs have no test sets
            graphs, labels = helper.group_same_size(self.test_graphs, self.test_labels)
            graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
            self.num_iterations_test = len(graphs)
            self.test_graphs_batches, self.test_labels_batches = graphs, labels


if __name__ == '__main__':
    config = utils.config.process_config('../configs/10fold_config.json')
    data = DataGenerator(config)
    data.initialize('train')


