import random

import numpy
import numpy as np
import torch
from torch.autograd import Variable


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    return_variable = False
    if isinstance(indexes, Variable):
        return_variable = True
        indexes = indexes.data
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    if return_variable:
        output = Variable(output, requires_grad=False)

    return output


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)


def prettyformat_dict_string(d, indent=''):
    result = list()
    for k, v in d.items():
        if isinstance(v, dict):
            result.append('{}{}:\t\n{}'.format(indent, k, prettyformat_dict_string(v, indent + '  ')))
        else:
            result.append('{}{}:\t{}\n'.format(indent, k, v))
    return ''.join(result)


def pack_list_of_lists(lol):
    offsets = list()
    ent_list = list()
    offsets.append(0)
    for l in lol:
        if isinstance(l, list) or isinstance(l, tuple):
            ent_list.extend(l)
            offsets.append(len(ent_list))
        else:
            ent_list.append(l)
            offsets.append(len(ent_list))
    offsets.append(-len(offsets)-1)
    out = (numpy.array(offsets)+len(offsets)).tolist()
    return out + ent_list

def unpack_list_of_lists(ents):
    ent_list = list()
    end = -1
    all_begin = -1
    all_end = -1
    for off in ents:
        if all_begin == -1:
            all_begin = off
        if off == 0:
            break
        if end == -1:
            end = off
            continue
        else:
            begin = end
        end = off
        all_end = off
        ent_list.append(ents[begin:end].tolist())
    return ent_list, ents[all_begin:all_end].tolist()


def argparse_bool_type(v):
    "Type for argparse that correctly treats Boolean values"
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")