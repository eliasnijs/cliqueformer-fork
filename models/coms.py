import math 
import torch 
import torch.nn as nn 
import torch.optim as optim 

from torch.autograd import Variable
from architectures.backbones import MLP 


class COMs(nn.Module):

    def __init__(self, input_dim, hidden_dims=2*(256,), lr=3e-4, gap=0.5, n_adversarial=50, lr_adversarial=5e-2, lr_alpha=1e-2):

        super().__init__()

        self.input_dim = input_dim
        self.target_regressor = MLP(input_dim, 1, hidden_dims, nn.LeakyReLU(0.3))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)   
        
        self.gap = gap 
        self.n_adversarial = n_adversarial
        self.lr_adversarial = lr_adversarial
        self.lr_alpha = lr_alpha
        self.log_alpha = Variable(torch.tensor([0.])).requires_grad_()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        return self.target_regressor(x)
        
    def adversarial_example(self, x, y):

        argmax = torch.argmax(y).item()
        x_adv = Variable(x[argmax:argmax+1]).requires_grad_().to(x.device)
        optim_adv = optim.Adam([x_adv], lr=self.lr_adversarial * math.sqrt(self.input_dim))

        for inner_step in range(self.n_adversarial):
            adv_val = self.forward(x_adv)
            (-adv_val).backward(retain_graph=True)
            optim_adv.step()
            optim_adv.zero_grad()
            self.optimizer.zero_grad()

        return x_adv.detach()


    def training_step(self, x, y):
        
        
        x_adv = self.adversarial_example(x, y)
        pred_adv = self.forward(x_adv)

        pred = self.forward(x)
        mse = ((pred - y)**2).mean()

        overestimation = (pred_adv - pred - self.gap).mean()
        alpha = torch.exp(self.log_alpha.to(x.device))

        loss = mse + alpha * overestimation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

        alpha = torch.exp(self.log_alpha.to(x.device))
        loss_alpha = -alpha * overestimation.detach()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        self.alpha_optimizer.zero_grad()

        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'overestimation': overestimation.item()
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


    
class COMsDiscrete(COMs):

    def __init__(self, n_input, input_dim, hidden_dims=2*(256,), lr=3e-4, gap=0.5, n_adversarial=50, lr_adversarial=5e-2, lr_alpha=1e-2):
        
        effective_input_dim = n_input * input_dim
        
        super().__init__(effective_input_dim, hidden_dims, lr, gap, n_adversarial, lr_adversarial, lr_alpha)

        self.n_input = n_input
        self.input_dim = input_dim
        self.effective_input_dim = effective_input_dim

    def forward(self, x):

        B, T, D = x.shape 
        x = x.reshape(B, -1)
        return self.target_regressor(x)