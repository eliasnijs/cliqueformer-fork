import torch 
import torch.nn as nn 
import torch.optim as optim 
import math 


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim = (256,), act=nn.ReLU()):

        super().__init__()
        
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.act = act 

        self.model = nn.Sequential()
        hidden_dim = (input_dim,) + hidden_dim 

        for i in range(len(hidden_dim)):

            self.model.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.model.append(act)
        
        self.model.append(nn.Linear(hidden_dim[-1], output_dim))
    
    def forward(self, x):

        return self.model(x)


class TransformerBlock(nn.Module):

    def __init__(self, model_dim, n_heads=2, dropout_rate=0.1, act=nn.GELU()):

        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate 
        self.act = act 

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.attention = nn.MultiheadAttention(model_dim, n_heads, batch_first=True)

        self.lin1 = nn.Linear(model_dim, 2 * model_dim)
        self.lin2 = nn.Linear(2 * model_dim, model_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, emb=None, extra_emb=None):

        x = x + emb if emb is not None else x 

        att, _= self.attention(x, x, x)
        x = self.norm1(x + att)

        x = x + extra_emb if extra_emb is not None else x

        x_mlp = self.lin2(self.act(self.lin1(x)))
        x = self.norm2(x + x_mlp)

        return x 


class Embedder(nn.Module):

    def __init__(self, dim: int):

        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(1, dim)
    
    def forward(self, x):
        x = x[..., None]
        return self.lin(x)


class Projector(nn.Module):

    def __init__(self, dim: int):

        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.lin(x).squeeze(-1)
        return x
