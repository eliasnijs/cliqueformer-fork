import ml_collections
from ml_collections.config_dict import config_dict
import torch 
import torch.nn as nn 

def get_config():

    config = ml_collections.ConfigDict()

    config.data = {
        'cls': 'Superconductor'
    }

    config.model = {
        'cls': 'COMs',
        'hidden_dims': 2 * (2048,),
        'n_adversarial': 50,
        'lr': 1e-3,
        'lr_adversarial': 5e-2,
        'lr_alpha': 1e-2
    }

    config.learner = {
        'cls': 'GradientAscent',
        'design_steps': 50,
        'decay': 0.,
        'lr': 5e-2
    }

    return config 