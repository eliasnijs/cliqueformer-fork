import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import architectures.backbones as backbones
import architectures.blocks as blocks 
import architectures.extras as extras 


class Transformer(nn.Module):

    def __init__(self, input_dim, transformer_dim, n_blocks=2, n_heads=2, p=0.1, act=nn.GELU()):

        super().__init__()

        self.embedder = blocks.Embedder(transformer_dim)
        self.bnorm_emb = nn.BatchNorm1d(input_dim)
        
        self.dim_embedding = extras.sinusoidal_embedding(input_dim, transformer_dim)
        self.dim_mlp = backbones.MLP(transformer_dim, transformer_dim, 1 * (2 * transformer_dim,))
        
        self.transformer = backbones.Transformer(transformer_dim, n_blocks, n_heads, p, act)

        self.act = act
        self.proj = blocks.Projector(transformer_dim)
        self.lin = nn.Linear(input_dim, 1)

    def forward(self, x):

        x = self.embedder(x)
        x = self.act(x)
        x = self.bnorm_emb(x)

        dim_emb = self.dim_mlp(self.dim_embedding.to(x.device))

        x = self.transformer(x, dim_emb)
        x = self.proj(x)
        x = self.act(x)
        x = self.lin(x)

        return x 
    

class MBOTransformer(nn.Module):

    def __init__(self, input_dim, transformer_dim, n_blocks=2, n_heads=2, p=0.1, act=nn.GELU(), lr=3e-4):

        super().__init__()

        self.target_regressor = Transformer(input_dim, transformer_dim, n_blocks, n_heads, p, act)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.target_regressor(x).squeeze(-1)
    
    def training_step(self, x, y):

        mse = (self.forward(x) - y)**2
        mse = mse.mean()
        mse.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            'mse': mse.item()
        }
    
    def eval_step(self, x, y):

        self.eval()

        with torch.no_grad():
            y_hat = self.forward(x)

        mse = ((y - y_hat)**2).mean()
        
        self.train()

        return {
            'mse': mse.item()
        }


class MBOTransformerDiscrete(MBOTransformer):

    def __init__(self, n_input, input_dim, transformer_dim, n_blocks=2, n_heads=2, p=0.1, act=nn.GELU()):
        
        effective_input_dim = n_input * input_dim

        super().__init__(effective_input_dim, transformer_dim, n_blocks, n_heads, p, act)   

    def forward(self, x):

        B, T, D = x.shape 
        x = x.reshape(B, -1)
        return super().forward(x)