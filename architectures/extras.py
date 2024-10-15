import torch 
import math 



def get_device(model):
    return list(model.parameters())[0].device


def sinusoidal_embedding(horizon, embedding_dim):

    steps = torch.arange(horizon)

    half_dim = embedding_dim//2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp( (-emb) * torch.arange(half_dim))

    emb = steps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)[None, ...]

    return emb


def fast_polyak(updatable_model, new_model, tau=0.005):

    one = torch.ones(1, requires_grad=False).to(get_device(updatable_model))

    for param, target_param in zip(
        new_model.parameters(), updatable_model.parameters()
    ):
        target_param.data.mul_(1 - tau)
        target_param.data.addcmul_(param.data, one, value=tau)


def rank(x, dim):

    x = torch.argsort(x, dim=dim)
    x = torch.argsort(x, dim=dim)

    return x.float()


def standardize(x, dim):

    x = x - torch.mean(x, dim=dim, keepdim=True)
    x = x / torch.std(x, dim=dim, keepdim=True)
    return x 


def center(x, dim):
    x_range = x.shape[dim] - 1
    x = x/x_range
    x -= 0.5
    return x 