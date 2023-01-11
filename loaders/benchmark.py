import torch
import data_benchmarking_gnns.data_helper as helper


class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, num_fold):
        self.dataset_name =  dataset_name
        self.num_fold = num_fold
        self.load_data()
        self.make_dataset()

    def load_data(self):
        graphs, labels = helper.load_dataset(self.dataset_name)
        if self.num_fold is None:
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = [graphs[i] for i in range(idx, len(graphs))], 
            [labels[i] for i in range(idx,len(graphs))], [graphs[i] for i in range(idx)], [labels[i] for i in range(idx)]
        elif self.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = [graphs[i] for i in train_idx], [labels[i] for i in train_idx], [graphs[i] for  i in test_idx], [labels[i] for i in test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.num_fold, self.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = [graphs[i] for i in train_idx], [labels[i] for i in train_idx], [graphs[i] for  i in test_idx], [labels[i] for i in test_idx]
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def make_dataset(self):
        self.train = [(torch.as_tensor(g, dtype=torch.float), torch.tensor(l, dtype=torch.long)) for (g,l) in zip(self.train_graphs, self.train_labels)] 
        self.val = [(torch.as_tensor(g, dtype=torch.float), torch.tensor(l, dtype=torch.long)) for (g,l) in zip(self.val_graphs, self.val_labels)]
