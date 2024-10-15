import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
import architectures.extras as extras
import distributions.gaussian as gauss

class Learner(nn.Module):

    def __init__(self, design, model, **kwargs):

        super().__init__()
        self.design = design 
        self.model = model 
        self.model.eval()

        decay = kwargs.pop("decay")
        self.lr = kwargs.pop('lr') * math.sqrt(design.param.shape[-1])

        if 'sgd' in kwargs and kwargs.pop('sgd'):
            effective_decay = decay / self.lr
            self.optimizer = optim.SGD([self.design.param], lr=self.lr, weight_decay=effective_decay)

        else:
            self.optimizer = optim.AdamW([self.design.param], lr=self.lr, weight_decay=decay) 

        if 'structure_fn' in kwargs:
            self.structure_fn = kwargs.pop('structure_fn')
        else:
            self.structure_fn = (lambda x: x)

    def value(self):
        x = self.structure_fn(self.design.param)
        return self.model(x).mean()
    
    def train_step(self):
        pass 

    def design_fn(self):
        return self.design.param

class GradientAscent(Learner):

    def __init__(self, design, model, **kwargs):
        super().__init__(design, model, **kwargs)

    def train_step(self):

        loss = -self.value()
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return {
            'loss': loss
        }


class GradientAscentDiscrete(GradientAscent):

    def __init__(self, design, model, **kwargs):
        
        super().__init__(design, model, **kwargs)

        n_input, input_dim = self.design.param.shape[-2:]
        self.n_input = n_input
        self.input_dim = input_dim
        self.structure_fn = (lambda x: x.view(-1, n_input * input_dim))

    def design_fn(self):
        x = torch.argmax(self.design.param, dim=-1)
        x = F.one_hot(x, self.input_dim)
        return x


class RWR(GradientAscent):

    def __init__(self, design, model, **kwargs):

        super().__init__(design, model, **kwargs)

        self.log_sigma = Variable(torch.zeros_like(design.param), requires_grad=True).to(design.param.device)
        self.temp = kwargs.pop("temp")
        self.optimizer = optim.Adam([self.design.param, self.log_sigma], lr=self.lr)

    def train_step(self):

        x_rwr, log_prob = gauss.from_params(self.design.param, self.log_sigma)
        
        x_input = self.structure_fn(x_rwr)
        val = self.model(x_input).reshape(-1).detach()
        
        weight = torch.clamp(torch.exp(val / self.temp), 1e-6, 1e3)
        loss = -(weight.reshape(-1) * log_prob.reshape(-1)).mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            'loss': loss
        }



class RWRDiscrete(RWR):

    def __init__(self, design, model, **kwargs):

        super().__init__(design, model, **kwargs)
        n_input, input_dim = self.design.param.shape[-2:]
        self.n_input = n_input
        self.input_dim = input_dim
        self.structure_fn = (lambda x: x.reshape(-1, n_input * input_dim))
    
    def design_fn(self):
        x = torch.argmax(self.design.param, dim=-1)
        x = F.one_hot(x, self.input_dim)
        return x

    def train_step(self):

        x_rwr, log_prob = gauss.from_params(self.design.param, self.log_sigma)
        log_prob = log_prob.sum(-1)

        x_input = self.structure_fn(x_rwr)
        val = self.model(x_input).reshape(-1).detach()
        
        weight = torch.clamp(torch.exp(val / self.temp), 1e-6, 1e2)
        weight = weight / weight.mean()
        loss = -(weight.reshape(-1) * log_prob.reshape(-1)).mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            'loss': loss
        }


