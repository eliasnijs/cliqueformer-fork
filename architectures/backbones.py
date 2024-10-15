import torch 
import torch.nn as nn 
import torch.optim as optim 
import architectures.blocks as blocks  
import architectures.extras as extras 
import distributions.gaussian as gauss 
import distributions.categorical as categorical

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, act = nn.GELU()):
        super().__init__()

        self.model = nn.Sequential()
        dims = (input_dim,) + hidden_dims

        for i in range(len(hidden_dims)):

            self.model.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]), act
                )
            )
        
        self.model.append(nn.Linear(dims[-1], output_dim))
    
    def forward(self, x):
        return self.model(x).squeeze(-1)
    


class Transformer(nn.Module):

    def __init__(self, model_dim, n_blocks=2, n_heads=2, dropout_rate=0.1, act=nn.GELU()):

        super().__init__()

        self.blocks = nn.Sequential()

        for i in range(n_blocks):

            self.blocks.append(
                blocks.TransformerBlock(model_dim, n_heads, dropout_rate, act)
            )
        
    def forward(self, x, emb=None):

        for i, block in enumerate(self.blocks):
            x = block(x, (emb if i == 0 else None))
        
        return x 
    

class DMLP(nn.Module):

    def __init__(self, n_input, input_dim, hidden_dims, dropout_rate= 0.1, act=nn.GELU()):

        super().__init__()
        self.n_input = n_input
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims 
        self.layers = nn.Sequential()

        self.act = act 
        dims = (input_dim,) + hidden_dims 

        for i in range(len(self.hidden_dims)):
            
            layer = nn.Sequential(
                nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(self.n_input), act, nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
        
        self.proj = blocks.Projector(dims[-1])

        h = self.hidden_dims[0]
        self.clique_emb = extras.sinusoidal_embedding(n_input, h)
        self.clique_mlp = MLP(h, h, (2 * h,), nn.ReLU())

    
    def forward(self, x):

        clique_emb = self.clique_mlp(self.clique_emb.to(x.device))

        for i, layer in enumerate(self.layers):

            x = layer(x)
            x = x + clique_emb if i == 0 else x 
        
        x = self.proj(x).mean(-1)

        return x 


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dims, beta=1., act=nn.GELU()):

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = MLP(input_dim, 2 * latent_dim, hidden_dims, act)
        self.decoder = MLP(latent_dim, 2 * input_dim, hidden_dims, act)
        self.beta = beta

    def encoder_fn(self, x):
        params = self.encoder(x)
        return params[..., :self.latent_dim], params[..., self.latent_dim:]

    def decoder_fn(self, z):
        params = self.decoder(x)
        return params[..., :self.input_dim], params[..., self.input_dim:]

    def encode(self, x):
        mu, _ = self.encoder_fn(x)
        return mu

    def decode(self, z):
        mu, _ = self.decoder_fn(z)
        return mu
    
    def posterior(self, x):
        mu, log_sigma = self.encoder_fn(x)
        sigma = torch.clamp(torch.exp(log_sigma), 1e-3, 1e2)
        noise = torch.randn(x.shape[0], self.latent_dim).to(x.device)
        
        z = mu + sigma * noise 
        post_log_lik = gauss.log_likelihood(mu, sigma, z)

        return z, post_log_lik, mu, sigma 
    
    def vae(self, x):

        z, _, mu, sigma = self.posterior(x)
        kl = gauss.standard_kl(mu, sigma)

        x_mu, x_log_sigma = self.decoder_fn(z)
        x_sigma = torch.clamp(torch.exp(x_log_sigma), 1e-3, 1e2)

        log_lik = gauss.log_likelihood(x_mu, x_sigma, x)

        rec = ((x_mu - x)**2).mean()
        std = x_sigma.mean()

        return z, mu, kl.mean(), log_lik.mean(), rec, std
    