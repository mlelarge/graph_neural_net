""" Masked tensors to handle batches with mixed node numbers """

import itertools
import functools
import torch
import torch.nn.functional as F

def from_list(tensor_list, dims, batch_name='B', base_name='N'):
    """
    Build a masked tensor from a list of tensors
    Dims is a tuple of dimensions which should be masked
    The tensors are supposed to agree on the other dimensions (and dtype)
    """
    dims = list(dims)
    n_dim = len(tensor_list[0].size())
    batch_size = len(tensor_list)

    # Create names
    data_names = [None] * (n_dim + 1)
    data_names[0] = batch_name
    for i, dim in enumerate(dims):
        data_names[dim+1] = base_name + i * '_'

    # Compute sizes of data and mask
    data_size = [0] * (n_dim + 1)
    data_size[0] = batch_size
    for dim in range(n_dim):
        data_size[dim+1] = max((tens.size(dim) for tens in tensor_list))

    # Fill data using padding
    data = torch.zeros(data_size, names=data_names, dtype=tensor_list[0].dtype)
    for i, tens in enumerate(tensor_list):
        # caution: dims for pad are specified from last to first
        data_padding = [[0, data_size[dim+1] - tens.size(dim)] for dim in range(n_dim)]
        data_padding = reversed(data_padding)
        data_padding = list(itertools.chain.from_iterable(data_padding))
        data[i] = F.pad(tens, data_padding)

    # Build mask
    mask = {}
    for dim, name in enumerate(data.names):
        if dim >= 1 and name:
            mask[name] = torch.zeros((batch_size, data.size(name)),
                                     names=(batch_name, name), dtype=data.dtype)
            for i, tens in enumerate(tensor_list):
                mask[name][i, :tens.size(dim-1)] = 1

    return MaskedTensor(data, mask, adjust_mask=False, apply_mask=False)

class MaskedTensor:
    """
    Masked tensor class
    - Unless you know what you are doing, should not be created with __init__,
      use from_list instead
    - Mask is always copied; data is copied iff copy is set to True
    - Individual tensors of a masked tensor mt can be retrived using list(mt),
      iterating with for tensor in mt or with indexing mt[i]
    """
    def __init__(self, data, mask, adjust_mask=True, apply_mask=False, copy=False, batch_name='B'):
        self.tensor = torch.tensor(data) if copy else data
        self.mask_dict = mask.copy()
        self._batch_name = batch_name
        if adjust_mask:
            self._adjust_mask_()
        if apply_mask:
            self.mask_()

    def __repr__(self):
        return "Data:\n{}\nMask:\n{}".format(self.tensor, self.mask_dict)

    ## Mask methods
    def _adjust_mask_(self):
        """ Check compatibily and remove unecessary masked dims """
        # To prevent changing the iterator during iteration
        mask_keys = list(self.mask_dict.keys())
        for name in mask_keys:
            mask_size = self.mask_dict[name].size(name)
            try:
                data_size = self.tensor.size(name)
                assert mask_size == data_size
            except RuntimeError:
                del self.mask_dict[name]

    def mask_(self):
        """ Mask data in place"""
        for mask in self.mask_dict.values():
            self.tensor = self.tensor * mask.align_as(self.tensor)

    def mask(self):
        """ Return new MaskedTensor with masked adata """
        return MaskedTensor(self.tensor, self.mask_dict, adjust_mask=False,
                            apply_mask=True, copy=True)

    ## Torch function override
    def __torch_function__(self, func, args=(), kwargs=None):
        """
        Support torch.* functions, derived from pytorch doc
        See https://pytorch.org/docs/master/notes/extending.html
        """
        if kwargs is None:
            kwargs = {}
        if func in SPECIAL_FUNCTIONS:
            return SPECIAL_FUNCTIONS[func](*args, **kwargs)
        new_args = [a.tensor if isinstance(a, MaskedTensor) else a for a in args]
        masks = (a.mask_dict for a in args if isinstance(a, MaskedTensor))
        new_mask = dict(item for mask_dict in masks for item in mask_dict.items())
        ret = func(*new_args, **kwargs)
        return MaskedTensor(ret, new_mask, adjust_mask=True, apply_mask=True)

    ## Iterator methods
    def __getitem__(self, index):
        item = self.tensor[index]
        names = item.names
        for dim, name in enumerate(names):
            if name:
                length = int(torch.sum(self.mask_dict[name][index]).item())
                item = torch.narrow(item, dim, 0, length)
        return item.rename(None)

    def __len__(self):
        return self.tensor.size(self._batch_name)

    def __iter__(self):
        return (self.__getitem__(index) for index in range(self.__len__()))

    ## Tensor methods
    def size(self, *args):
        """ Return size of the underlying tensor """
        return self.tensor.size(*args)

    def permute(self, *dims):
        """ Permute the tensor """
        # Unfortunately, permute is not yet implemented for named tensors
        # So we do it by hand
        if len(dims) != len(self.tensor.size()):
            raise ValueError
        names = self.tensor.names
        nameless_tensor = self.tensor.rename(None).permute(*dims)
        permuted_names = [names[dim] for dim in dims]
        res_tensor = nameless_tensor.rename(*permuted_names)
        return MaskedTensor(res_tensor, self.mask_dict, adjust_mask=False, apply_mask=False)

    def to(self, *args, **kwargs):
        """ Apply the method .to() to both tensor and mask """
        new_dict = {name:mask.to(*args, **kwargs) for name, mask in self.mask_dict.item()}
        new_tensor = self.tensor.to(*args, **kwargs)
        return MaskedTensor(new_tensor, new_dict, adjust_mask=False, apply_mask=False)

### Torch function overrides
SPECIAL_FUNCTIONS = {}

def implements(torch_function):
    """
    Register a torch function override for MaskedTensor
    See https://pytorch.org/docs/master/notes/extending.html
    """
    @functools.wraps(torch_function)
    def decorator(func):
        SPECIAL_FUNCTIONS[torch_function] = func
        return func
    return decorator

def get_dtype_min_value(dtype):
    """ Get the min value of given dtype, whether int or float """
    try:
        return torch.finfo(dtype).min
    except TypeError:
        pass
    try:
        return torch.iinfo(dtype).min
    except TypeError:
        raise TypeError("dtype is neither float nor int")

@implements(torch.max)
def torch_max(masked_tensor, dim):
    """ Implements torch.max """
    tensor = masked_tensor.tensor
    min_value = get_dtype_min_value(tensor.dtype)
    for mask in masked_tensor.mask_dict.values():
        aligned_mask = mask.align_as(tensor)
        tensor = tensor * aligned_mask + min_value * (1 - aligned_mask)
    max_tensor, indices = torch.max(tensor, dim)
    new_masked_tensor = MaskedTensor(max_tensor, masked_tensor.mask_dict,
                                     adjust_mask=True, apply_mask=True)
    return new_masked_tensor, indices

@implements(F.conv2d)
def torch_conv2d(inp, *args, **kwargs):
    """ Implements conv2d on masked tensors """
    # Unfortunately, conv2d does not support named tensors yet
    names = inp.tensor.names
    nameless_tensor = inp.tensor.rename(None)
    nameless_res_tensor = F.conv2d(nameless_tensor, *args, **kwargs)
    res_tensor = nameless_res_tensor.rename(*names)
    return MaskedTensor(res_tensor, inp.mask_dict, adjust_mask=False, apply_mask=True)
