import torch
import torch.nn as nn
from collections import defaultdict

#####################
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)  

def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k,v in nested_dict.items()}

def group_by_key(items):
    res = defaultdict(list)
    for k, v in items: 
        res[k].append(v) 
    return res

#####################
## graph building
#####################
sep = '/'

def split(path):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]

def normpath(path):
    #simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == '..': parts.pop()
        elif p.startswith(sep): parts = [p]
        else: parts.append(p)
    return sep.join(parts)

has_inputs = lambda node: type(node) is tuple

def pipeline(net):
    return [(sep.join(path), (node if has_inputs(node) else (node, [-1]))) for (path, node) in path_iter(net)]

def build_graph(net):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: normpath(sep.join((path, '..', rel_path))) if isinstance(rel_path, str) else flattened[idx+rel_path][0]
    return {path: (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]]) for idx, (path, node) in enumerate(flattened)}

class Network(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.graph = build_graph(net)
        for path, (val, _) in self.graph.items(): 
            setattr(self, path.replace('/', '_'), val)
    
    def nodes(self):
        return (node for node, _ in self.graph.values())
    
    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (node, ins) in self.graph.items():
            #only compute nodes that are not supplied as inputs.
            if k not in outputs: 
                outputs[k] = node(*[outputs[x] for x in ins])
        return outputs
    
    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self