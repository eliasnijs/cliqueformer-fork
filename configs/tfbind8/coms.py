import ml_collections
from ml_collections.config_dict import config_dict
import torch 
import torch.nn as nn 

def get_config():

    config = ml_collections.ConfigDict()

    config.data = {
        'cls': 'TFBind8'
    }

    config.model = {
        'cls': 'COMsDiscrete',
        'hidden_dims': 2 * (2048,),
        'n_adversarial': 50,
        'lr': 1e-3,
        'lr_adversarial': 2,
        'lr_alpha': 1e-2
    }

    config.learner = {
        'cls': 'GradientAscentDiscrete',
        'design_steps': 50,
        'decay': 0.,
        'lr': 2
    }

    return config 