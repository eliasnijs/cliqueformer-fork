import ml_collections
from ml_collections.config_dict import config_dict
import torch 
import torch.nn as nn 

def get_config():

    config = ml_collections.ConfigDict()

    config.data = {
        'cls': 'LRBF',
        'N': int(1e5),
        'd': 41 
    }

    config.model = {
        'cls': 'MBOTransformer',
        'transformer_dim': 64,
        'n_blocks': 2, 
        'n_heads': 2,
        'hidden_dims': 2 * (256,),
        'p_tran': 0.5, 
        'act': nn.GELU(),
        'lr': 1e-4 
    }

    config.learner = {
        'cls': 'GradientAscent',
        'design_steps': 50,
        'decay': 0.,
        'lr': 5e-2,
    }

    return config 


