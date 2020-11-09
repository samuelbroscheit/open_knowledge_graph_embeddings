from collections import OrderedDict

import torch


class Permute(torch.nn.Module):

    def __init__(self, *to_dims):
        super(Permute, self).__init__()
        self.to_dims = to_dims

    def forward(self, input: torch.FloatTensor):
        return input.permute(self.to_dims).contiguous()


class View(torch.nn.Module):

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input: torch.FloatTensor):
        return input.view(self.shape)

class Sequential(torch.nn.Sequential):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            discount_none = 0
            for idx, module in enumerate(args):
                if module:
                    self.add_module(str(idx-discount_none), module)
                else:
                    discount_none += 1

    def forward(self, *args, **kwargs):
        for i, module in enumerate(self._modules.values()):
            if i == 0:
                result = module(*args, **kwargs)
            else:
                result = module(result)
        return result
