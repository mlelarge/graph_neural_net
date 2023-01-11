import maskedtensors.maskedtensor as maskedtensor
from torch.utils.data import DataLoader#, default_collate
import torch

def collate_fn_pair(samples_list):
    input1_list = [input1 for input1, _ in samples_list]
    input2_list = [input2 for _, input2 in samples_list]
    input1 = maskedtensor.from_list(input1_list, dims=(1, 2), base_name='N')
    input2 = maskedtensor.from_list(input2_list, dims=(1, 2), base_name='M')
    return input1, input2

def collate_fn_pair_explore(samples_list):
    input1_list = [input1 for input1, _ in samples_list]
    input2_list = [input2 for _, input2 in samples_list]
    return {'input': torch.stack(input1_list)}, {'input': torch.stack(input2_list)}

def siamese_loader(data, batch_size, constant_n_vertices, shuffle=True):
    assert len(data) > 0
    if constant_n_vertices:
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=4, collate_fn=collate_fn_pair_explore)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=4, collate_fn=collate_fn_pair)

def collate_fn(samples_list):
    inputs = [inp for inp,_ in samples_list]
    labels = [lab for _,lab in samples_list]
    return maskedtensor.from_list(inputs, dims=(1, 2), base_name='N'), torch.tensor(labels)


def collate_fn_explore(samples_list):
    graphs = [inp[0,:,:].unsqueeze(0) for inp,_ in samples_list]
    nodes_f = [torch.diagonal(inp[1:,:,:], dim1=1, dim2=2) for inp,_ in samples_list]
    labels = [lab for _,lab in samples_list]
    #print(nodes_f)
    return {'graphs': maskedtensor.from_list(graphs, dims=(1, 2), base_name='N'),
            'nodes_f': maskedtensor.from_list(nodes_f, dims=(1,), base_name='N'),
            'target': torch.tensor(labels)}
  

def simple_loader(data, batch_size, constant_n_vertices, shuffle=True):
    assert len(data) > 0
    if constant_n_vertices:
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=4)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=0, collate_fn=collate_fn_explore)

def collate_fn_benchmark(list):
    graphs = [inp[0,:,:].unsqueeze(0) for inp,_ in list]
    nodes_f = [torch.diagonal(inp[1:,:,:], dim1=1, dim2=2) for inp,_ in list]
    labels = [lab for _,lab in list]
    #print(nodes_f)
    return {'graphs': maskedtensor.from_list(graphs, dims=(1, 2), base_name='N'),
            'nodes_f': maskedtensor.from_list(nodes_f, dims=(1,), base_name='N'),
            'target': torch.tensor(labels)}

def benchmark_loader(data, batch_size, constant_n_vertices=False, shuffle=True):
    assert len(data) > 0
    if constant_n_vertices:
        print('Not implemented')
        #return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=0, collate_fn=collate_fn_benchmark)