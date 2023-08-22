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
    n_dim = tensor_list[0].dim()
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
        #self._is_cuda = self.tensor.is_cuda
        self.dtype = self.tensor.dtype
        self.device = self.tensor.device
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
    @classmethod
    def __torch_function__(self, func, types, args=(), kwargs=None):
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
    
    def dim(self):
        return self.tensor.dim()

    def contiguous(self, *args):
        self.tensor = self.tensor.contiguous(*args)
        return self

    def view(self, *dims):
        """ only acting on named dim None which should be at the end
        i.e. not acting on masked dimensions or batch dimension"""
        names = self.tensor.names
        nameless_tensor = self.tensor.rename(None).view(*dims)
        new_names = [None] * len(dims)
        for (i,n) in enumerate(names):
            if  i < len(dims):
                new_names[i] = n
        res_tensor = nameless_tensor.rename(*new_names)
        return MaskedTensor(res_tensor, self.mask_dict, adjust_mask=False, apply_mask=False)

    @property
    def shape(self):
        """ Return shape of the underlying tensor """
        return self.tensor.size()

    @property
    def is_cuda(self):
        return self.tensor.is_cuda

    @property
    def get_device(self):
        return self.tensor.get_device()

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
        new_dict = {name:mask.to(*args, **kwargs) for name, mask in self.mask_dict.items()}
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
def torch_max(masked_tensor, dim=None):
    """ Implements torch.max """
    if dim is None:
        # !!! taking the max over the whole batch !!!
        return torch.max(masked_tensor.tensor)
    else:
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

@implements(F.linear)
def torch_linear(inp, *args, **kwargs):
    """ Implements linear on masked tensors """
    # Unfortunately, linear does not support named tensors yet
    names = inp.tensor.names
    nameless_tensor = inp.tensor.rename(None)
    nameless_res_tensor = F.linear(nameless_tensor, *args, **kwargs)
    res_tensor = nameless_res_tensor.rename(*names)
    return MaskedTensor(res_tensor, inp.mask_dict, adjust_mask=False, apply_mask=True)

@implements(torch.cat)
def torch_cat(tensors, dim=0):
    """
    Implements torch.cat for masked tensors
    We have to implement it manually for the same reason as the issue
    mentionned below
    """
    # Improvement: find a more elegant way when pytorch finds an elegant way
    # for the issues mentionned below
    new_args = [a.tensor if isinstance(a, MaskedTensor) else a for a in tensors]
    masks = (a.mask_dict for a in tensors if isinstance(a, MaskedTensor))
    new_mask = dict(item for mask_dict in masks for item in mask_dict.items())
    ret = torch.cat(new_args, dim=dim)
    return MaskedTensor(ret, new_mask, adjust_mask=False, apply_mask=False)

def dispatch_cat(tensors, dim=0):
    """
    Temporary workaround to dispatch issue with torch.cat
    See https://github.com/pytorch/pytorch/issues/34294
    """
    tensor = tensors[0]
    if isinstance(tensor, torch.Tensor):
        return torch.cat(tensors, dim=dim)
    return tensor.__torch_function__(torch.cat, [type(t) for t in tensors], (tensors,), {'dim':dim})

@implements(torch.stack)
def torch_stack(tensors, dim=0):
    """
    same pb as above
    """
    # Unfortunately, does not support named tensors yet...
    new_args = [a.tensor.rename(None) if isinstance(a, MaskedTensor) else a for a in tensors]
    names = [a.tensor.names if isinstance(a, MaskedTensor) else None for a in tensors]
    try:
        assert names[0] == names[1]
    except:
        print('trying to stack uncompatible masked tensors')
    masks = (a.mask_dict for a in tensors if isinstance(a, MaskedTensor))
    new_mask = dict(item for mask_dict in masks for item in mask_dict.items())
    ret = torch.stack(new_args, dim=dim)
    new_names = names[0][0:dim] + (None,) + names[0][dim:]
    return MaskedTensor(ret.refine_names(*new_names), new_mask, adjust_mask=True, apply_mask=False)

def dispatch_stack(tensors, dim=0):
    tensor = tensors[0]
    if isinstance(tensor, torch.Tensor):
        return torch.stack(tensors, dim=dim)
    return tensor.__torch_function__(torch.stack, [type(t) for t in tensors], (tensors,), {'dim':dim})


@implements(torch.flatten)
def torch_flatten(inp, start_dim=0, end_dim=-1):
    """ Implements torch.flatten """
    # Unfortunately, does not support named tensors yet...
    names = inp.tensor.names
    new_names = names[0:start_dim] + (None,) + names[end_dim+1:]
    res_tensor = torch.flatten(inp.tensor.rename(None), start_dim=start_dim, end_dim=end_dim)
    res_tensor = res_tensor.refine_names(*new_names)
    return MaskedTensor(res_tensor, inp.mask_dict, adjust_mask=True, apply_mask=False)

def get_sizes(masked_tensor, keepdim=False):
    # returns the number of non-masked entries
    full_mask = torch.ones_like(masked_tensor.tensor)
    names = tuple(masked_tensor.mask_dict.keys())
    for mask in masked_tensor.mask_dict.values():
        aligned_mask = mask.align_as(full_mask)
        full_mask = full_mask * aligned_mask 
    return torch.sum(full_mask, dim=names, keepdim=keepdim)

@implements(torch.mean)
def torch_mean(masked_tensor, keepdim = False, *args, **kwargs):
    # returns a tensor
    # args are not taken into account!
    # computing the mean over the masked dimensions
    sizes = get_sizes(masked_tensor, keepdim=keepdim)
    names = tuple(masked_tensor.mask_dict.keys())
    return torch.sum(masked_tensor.tensor, dim = names, keepdim=keepdim)/sizes

@implements(torch.var)
def torch_var(masked_tensor, keepdim = False, *args, **kwargs):
    # same restriction as above!
    sizes = get_sizes(masked_tensor, keepdim=keepdim)
    means = torch_mean(masked_tensor, keepdim=True)
    vars = MaskedTensor((masked_tensor.tensor - means)**2, masked_tensor.mask_dict, adjust_mask=False, apply_mask=True)
    names = tuple(vars.mask_dict.keys())
    return torch.sum(vars.tensor, dim = names, keepdim=keepdim)/sizes

@implements(F.instance_norm)
def torch_instance_norm(masked_tensor, eps=1e-05, weight=None, bias =None, *args, **kwargs):
    """ Implements instance_norm on masked tensors 
    only works for shape (b,f,n,n) when normalization is taken on (n,n) (InstanceNorm2d)
    for each feature f with track_running_stats=False
    """
    # Unfortunately, InstanceNorm2d does not support named tensors yet
    means = torch_mean(masked_tensor, keepdim=True)
    var_s = torch_var(masked_tensor, keepdim=True)
    res_tensor = (masked_tensor.tensor - means)/torch.sqrt(var_s+eps)
    if (weight is not None) and (bias is not None):
        res_tensor = weight.reshape(1,weight.shape[0],1,1)*res_tensor+bias.reshape(1,bias.shape[0],1,1)
    return MaskedTensor(res_tensor, masked_tensor.mask_dict, adjust_mask=False, apply_mask=True)

@implements(F.layer_norm)
def torch_layer_norm(masked_tensor, *args, **kwargs):
    """ 
    Implements layer_norm on masked tensors
    when applied accross channel direction (not accross masked dim!)
    https://github.com/pytorch/pytorch/issues/81985#issuecomment-1236143883
    """
    names = masked_tensor.tensor.names
    nameless_tensor = masked_tensor.tensor.rename(None)
    nameless_res_tensor = F.layer_norm(nameless_tensor, *args, **kwargs)
    res_tensor = nameless_res_tensor.rename(*names)
    return MaskedTensor(res_tensor, masked_tensor.mask_dict, adjust_mask=False, apply_mask=True)

@implements(torch.diag_embed)
def torch_diag_embed(inp, offset=0, dim1=-2, dim2=-1, *args, **kwargs):
    names = inp.tensor.names
    new_name = names[-1]+'_'
    new_names = names + (new_name,)
    nameless_tensor = inp.tensor.rename(None)
    nameless_res_tensor = torch.diag_embed(nameless_tensor, offset=offset, dim1=dim1, dim2=dim2, *args, **kwargs)
    res_tensor = nameless_res_tensor.rename(*new_names)
    new_dict = inp.mask_dict
    new_mask = inp.mask_dict[names[-1]].rename(None)
    names_mask = inp.mask_dict[names[-1]].names[:-1] +(new_name,)
    new_dict[new_name] = new_mask.rename(*names_mask)
    return MaskedTensor(res_tensor, new_dict, adjust_mask=False, apply_mask=True)    

@implements(F.nll_loss)
def torch_nll_loss(masked_tensor, target, *args, **kwargs):
    return F.nll_loss(masked_tensor.tensor.rename(None), target, *args, **kwargs)

@implements(F.cross_entropy)
def torch_cross_entropy(masked_tensor, target, *args, **kwargs):
    return F.cross_entropy(masked_tensor.tensor.rename(None), target, *args, **kwargs)