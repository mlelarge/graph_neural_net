import torch
import torch.utils
import toolbox.maskedtensor as maskedtensor

def collate_fn(samples_list):
    input1_list = [input1 for input1, _ in samples_list]
    input2_list = [input2 for _, input2 in samples_list]
    input1 = maskedtensor.from_list(input1_list, dims=(0, 1), base_name='N')
    input2 = maskedtensor.from_list(input2_list, dims=(0, 1), base_name='M')
    return input1, input2

def siamese_loader(data, batch_size, constant_n_vertices, shuffle=True):
    assert len(data) > 0
    if constant_n_vertices and False:
        return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=4)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, collate_fn=collate_fn)
