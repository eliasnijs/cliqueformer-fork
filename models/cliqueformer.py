import math 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import architectures.backbones as backbones
import architectures.blocks as blocks 
import architectures.extras as extras 
import distributions.gaussian as gauss 
import distributions.categorical as categorical
import models.graphops as graphops
from copy import deepcopy


class CliqueformerEncoder(nn.Module):
  
    def __init__(self, input_dim, transformer_dim, n_cliques, clique_dim, overlap, n_blocks=2, n_heads=2, p=0.1, act=nn.GELU()):
    
        super().__init__()

        self.embedder = blocks.Embedder(transformer_dim)
        self.bnorm_emb = nn.BatchNorm1d(input_dim)
        
        self.dim_embedding = extras.sinusoidal_embedding(input_dim, transformer_dim)
        self.dim_mlp = backbones.MLP(transformer_dim, transformer_dim, 1 * (2 * transformer_dim,))

        self.transformer = backbones.Transformer(transformer_dim, n_blocks, n_heads, p, act)
        self.index_matrix = graphops.chain_of_cliques(n_cliques, clique_dim, overlap)
        self.latent_proj = blocks.Projector(transformer_dim)
        self.latent_dim = (clique_dim - overlap) * (n_cliques - 1) + clique_dim
        self.latent_lin = nn.Linear(input_dim, 2 * self.latent_dim)
        self.bnorm = nn.BatchNorm1d(input_dim)
        self.act = act 

        self.bnorm_emb = nn.BatchNorm1d(input_dim)
        self.bnorm_lat = nn.BatchNorm1d(input_dim)

    def forward(self, x, separate=True):
        
        x = self.embedder(x)
        x = self.act(x)
        x = self.bnorm_emb(x)
        dim_emb = self.dim_mlp(self.dim_embedding.to(x.device))

        x = self.transformer(x, dim_emb)
        x = self.latent_proj(x)
        x = self.act(x) ### Weirdly, it works better with act before bnorm. Everywhere else act after bnorm
        x = self.bnorm_lat(x)
        x = self.latent_lin(x)

        mu, log_sigma = x[..., :self.latent_dim], x[..., self.latent_dim:]

        if not separate:
            return mu, log_sigma
        
        index_matrix = self.index_matrix.to(x.device)
        mu = graphops.separate_latents(mu, index_matrix)
        log_sigma = graphops.separate_latents(log_sigma, index_matrix)

        return mu, log_sigma
    

class CliqueformerEncoderDiscrete(CliqueformerEncoder):

    def __init__(self, n_input, input_dim, transformer_dim, n_cliques, clique_dim, overlap, n_blocks=2, n_heads=2, p=0.1, act=nn.GELU()):
        
        super().__init__(input_dim, transformer_dim, n_cliques, clique_dim, overlap, n_blocks, n_heads, p, act)
        
        self.embedder = nn.Linear(input_dim, transformer_dim)
        self.dim_embedding = extras.sinusoidal_embedding(n_input, transformer_dim)
        self.bnorm_emb = nn.BatchNorm1d(n_input)
        self.bnorm_lat = nn.BatchNorm1d(n_input)
        self.latent_lin = nn.Linear(n_input, 2 * self.latent_dim)
        self.n_input = n_input


class CliqueformerDecoder(nn.Module):
   
    def __init__(self, n_cliques, clique_dim, output_dim, transformer_dim, n_blocks=2, n_heads=2, p=0.1, act=nn.GELU()):

        super().__init__()
        self.n_cliques = n_cliques
        self.clique_dim = clique_dim 
        self.output_dim = output_dim 
        self.transformer_dim = transformer_dim 
        self.n_blocks = n_blocks  
        self.act = act 
        
        self.lin = nn.Linear(clique_dim, transformer_dim)
        self.clique_embedding = extras.sinusoidal_embedding(n_cliques, transformer_dim)
        self.clique_mlp = backbones.MLP(transformer_dim, transformer_dim, 1 * (2 * transformer_dim,))

        self.transformer = backbones.Transformer(transformer_dim, n_blocks, n_heads, p, act)
        self.lin_reshape = nn.Linear(transformer_dim, clique_dim)
        self.lin_out = nn.Linear(n_cliques * clique_dim, 2* output_dim)

        self.bnorm_lin = nn.BatchNorm1d(n_cliques)
        self.bnorm_reshape = nn.BatchNorm1d(n_cliques)



    def forward(self, x):
        
        x = self.lin(x)
        x = self.act(x)
        x = self.bnorm_lin(x)
        clique_emb = self.clique_mlp(self.clique_embedding.to(x.device))

        x = self.transformer(x, clique_emb)
        x = self.lin_reshape(x)
        x = self.act(x)
        x = self.bnorm_reshape(x).reshape(-1, self.n_cliques * self.clique_dim) 
        params = self.lin_out(x)

        mu, log_sigma = params[..., : self.output_dim], params[..., self.output_dim : ]
        
        return mu, log_sigma 


class CliqueformerDecoderDiscrete(CliqueformerDecoder):

    def __init__(self, n_output, n_cliques, clique_dim, output_dim, transformer_dim, n_blocks=2, n_heads=2, p=0.1, act=nn.GELU()):
        
        super().__init__(n_cliques, clique_dim, output_dim, transformer_dim, n_blocks, n_heads, p, act)
        self.lin_reshape = nn.Linear(transformer_dim, n_output * output_dim)
        self.lin_out = nn.Linear(n_cliques *  output_dim, output_dim)
        self.n_output = n_output

    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)

        x = self.bnorm_lin(x) 
        clique_emb = self.clique_mlp(self.clique_embedding.to(x.device))

        x = self.transformer(x, clique_emb)
        x = self.lin_reshape(x) 
        x = self.act(x)
        x = self.bnorm_reshape(x).reshape(-1, self.n_output, self.n_cliques * self.output_dim)
        x = self.lin_out(x).reshape(-1, self.n_output, self.output_dim)
        
        return F.softmax(x, -1)



class Cliqueformer(nn.Module):

    def __init__(self, input_dim, n_cliques, clique_dim, overlap, transformer_dim=128, n_blocks=2, n_heads=2, hidden_dims=2 * (256,), 
                 p_tran = 0.1, p_mlp = 0.1, act = nn.GELU(), alpha_vae=1, beta_vae=0, temp_mse=10, lr=3e-4, polyak_tau=5e-3):

        super().__init__()
        self.input_dim = input_dim
        self.n_cliques = n_cliques
        self.clique_dim = clique_dim 
        self.overlap = overlap 
        self.transformer_dim = transformer_dim 
        self.latent_dim = (clique_dim - overlap) * (n_cliques - 1) + clique_dim

        self.encoder = CliqueformerEncoder(input_dim, transformer_dim, n_cliques, clique_dim, overlap, n_blocks, n_heads, p_tran, act)
        self.decoder = CliqueformerDecoder(n_cliques, clique_dim, input_dim, transformer_dim, n_blocks, n_heads, p_tran, act)
        self.regressor = backbones.DMLP(n_cliques, clique_dim, hidden_dims, p_mlp, act)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.target_regressor = deepcopy(self.regressor)

        self.index_matrix = self.encoder.index_matrix
        self.alpha_vae = alpha_vae 
        self.beta_vae = beta_vae
        self.temp_mse = temp_mse
        self.polyak_tau = polyak_tau

    def posterior(self, x):
        
        z_mu, z_log_sigma = self.encoder(x)
        z_sigma = torch.clamp(torch.exp(z_log_sigma), 1e-3, 1e2)
        noise = torch.randn(x.shape[0], self.latent_dim)
        noise = graphops.separate_latents(noise, self.index_matrix).to(x.device)

        z = z_mu + z_sigma * noise.to(x.device)
        post_log_lik = gauss.log_likelihood(z_mu, z_sigma, z)

        return z, post_log_lik, z_mu, z_sigma
    
    def encode(self, x, separate=True):
        z, _ = self.encoder(x, separate)
        return z 
    
    def decode(self, z):
        x, _ = self.decoder(z)
        return x 
    
    def predict(self, z):
        return self.regressor(z) 
    
    def vae(self, x):
        
        z, _, z_mu, z_sigma = self.posterior(x)
        kl = gauss.standard_kl(z_mu, z_sigma)
        indexes = torch.randint(0, kl.shape[1], (kl.shape[0], 1)).to(x.device)
        kl_rand = torch.gather(kl, 1, indexes)
        x_mu, x_log_sigma = self.decoder(z)
        x_sigma = torch.clamp(torch.exp(x_log_sigma), 1e-3, 1e2)

        log_lik = gauss.log_likelihood(x_mu, x_sigma, x)

        rec = ((x_mu - x)**2).mean()
        std = x_sigma.mean() 

        return z, z_mu, kl_rand.mean(), log_lik.mean(), rec, std 
    
    def training_step(self, x, y):
        
        z, z_mu, kl, log_lik, rec, std = self.vae(x)
        y_hat = self.predict(z) 

        mse = ((y - y_hat)**2).mean() 

        loss = self.alpha_vae * self.beta_vae * (self.temp_mse * mse + kl) - log_lik
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
        self.optimizer.zero_grad()

        extras.fast_polyak(self.target_regressor, self.regressor, self.polyak_tau)

        return {
            'loss': loss.item(),
            'kl': kl.item(),
            'log_lik': log_lik.item(),
            'rec': rec.item(),
            'std': std.item(),
            'mse': mse.item()
        }
    
    def eval_step(self, x, y):
        
        self.eval()

        with torch.no_grad():
            z, z_mu, kl, log_lik, rec, std = self.vae(x)
            y_hat = self.predict(z)
            
        mse = ((y - y_hat)**2).mean()
        loss = self.beta_vae * (self.temp_mse * mse + kl) - log_lik

        self.train()

        return {
            'loss': loss.item(),
            'kl': kl.item(),
            'log_lik': log_lik.item(),
            'rec': rec.item(),
            'std': std.item(),
            'mse': mse.item()
        }

        
    
class CliqueformerDiscrete(Cliqueformer):

    def __init__(self, n_input, input_dim, n_cliques, clique_dim, overlap, transformer_dim=128, n_blocks=2, n_heads=2, hidden_dims=2 * (256, ), 
                 p_tran=0.1, p_mlp=0.1, act=nn.GELU(), alpha_vae=1, beta_vae=0, temp_mse=10, lr=3e-4, polyak_tau=5e-3):
        
        super().__init__(input_dim, n_cliques, clique_dim, overlap, transformer_dim, n_blocks, n_heads, hidden_dims, 
                         p_tran, p_mlp, act, alpha_vae, beta_vae, temp_mse, lr, polyak_tau)
        self.n_input = n_input
        self.encoder = CliqueformerEncoderDiscrete(n_input, input_dim, transformer_dim, n_cliques, clique_dim, overlap, n_blocks, n_heads, p_tran, act)
        self.decoder = CliqueformerDecoderDiscrete(n_input, n_cliques, clique_dim,  input_dim, transformer_dim, n_blocks, n_heads, p_tran, act)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def decode(self, z):
        probs = self.decoder(z)
        x = torch.argmax(probs, dim=-1)
        x = F.one_hot(x, self.input_dim)
        return x 
    
    def vae(self, x):

        z, _, z_mu, z_sigma = self.posterior(x)
        kl = gauss.standard_kl(z_mu, z_sigma)

        indexes = torch.randint(0, kl.shape[1], (kl.shape[0], 1)).to(x.device)
        kl_rand = torch.gather(kl, 1, indexes)
        
        probs = self.decoder(z)
        x = torch.argmax(x, dim=-1, keepdim=True)
        log_lik = categorical.log_likelihood(probs, x)
        
        rec = 1 - torch.gather(probs, 2, x).mean()
        ent = -(probs * torch.log(probs + 1e-8)).sum(-1)
        std = (ent / math.log(self.input_dim)).mean()

        return z, z_mu, kl_rand.mean(), log_lik.mean(), rec, std 
