import math 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from architectures.backbones import MLP

class Naive(nn.Module):

    def __init__(self, input_dim, hidden_dims=2*(256,), lr=3e-4):
        
        super().__init__()
        self.target_regressor = MLP(input_dim, 1, hidden_dims, nn.ReLU())
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

    
class NaiveDiscrete(Naive):

    def __init__(self, n_input, input_dim, hidden_dims=2*(256,), lr=3e-4):

        effective_input_dim = n_input * input_dim

        super().__init__(effective_input_dim, hidden_dims, lr)
    
    def forward(self, x):

        B, T, D = x.shape 
        x = x.reshape(B, -1)
        return self.target_regressor(x)
