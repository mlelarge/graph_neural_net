import toolbox.maskedtensor as maskedtensor
from torch.utils.data import DataLoader
from models.gcn_model import data_to_dgl_format

def collate_fn(samples_list):
    input1_list = [input1 for input1, _ in samples_list]
    input2_list = [input2 for _, input2 in samples_list]
    input1 = maskedtensor.from_list(input1_list, dims=(0, 1), base_name='N')
    input2 = maskedtensor.from_list(input2_list, dims=(0, 1), base_name='M')
    return input1, input2

def siamese_loader(data, batch_size, constant_n_vertices, shuffle=True):
    assert len(data) > 0
    if constant_n_vertices:
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=4)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, collate_fn=collate_fn)

def get_loader(architecture: str, data_object: any, batch_size: int, constant_n_vertices: bool=True, shuffle: bool=True)->DataLoader:
    """This function creates the appropriate DataLoader depending on the architecture of the problem"""
    arch = architecture.lower()
    if arch == 'fgnn':
        return siamese_loader(data_object, batch_size, constant_n_vertices, shuffle)
    elif arch == 'gcn':
        data_object = data_to_dgl_format(data_object)
        return siamese_loader(data_object, batch_size, constant_n_vertices, shuffle)
    else:
        raise NotImplementedError(f"Architecture {arch} not implemented. Choose among 'fgnn' and 'gcn'.")
