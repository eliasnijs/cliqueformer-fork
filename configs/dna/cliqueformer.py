import ml_collections
from ml_collections.config_dict import config_dict
import torch 
import torch.nn as nn 

def get_config():

    config = ml_collections.ConfigDict()

    config.data = {
        'cls': 'DNA',
        'from_existing': False,
        'dna_property': 'k562'
    }

    config.model = {
        'cls': 'CliqueformerDiscrete',
        'n_cliques': 40, 
        'clique_dim': 6,
        'overlap': 1, 
        'transformer_dim': 256,
        'n_blocks': 2, 
        'n_heads': 2,
        'hidden_dims': 2 * (512,),
        'p_tran': 0.5, 
        'p_mlp': 0.5, 
        'alpha_vae': 1,
        'temp_mse': 10,
        'act': nn.GELU(),
        'lr': 1e-4 
    }

    config.learner = {
        'cls': 'GradientAscent',
        'design_steps': 1000,
        'decay': 0.5,
        'lr': 3e-4
    }

    return config 