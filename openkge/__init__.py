import torch


def tprint(t):
    if isinstance(t, torch.Tensor):
        print(t, '\n', t.shape, t.dtype, t.device)
    else:
        print (t)