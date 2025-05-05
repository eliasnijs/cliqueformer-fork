import ml_collections
from ml_collections.config_dict import config_dict
import torch
import torch.nn as nn

def get_config():

    config = ml_collections.ConfigDict()

    config.data = {
        'cls': 'DNA',
        'from_existing': False,
        'dna_property': 'hepg2'
    }

    config.model = {
        'cls': 'MBOTransformerDiscrete',
        'transformer_dim': 32,  # Reduced from 64
        'n_blocks': 1,          # Reduced from 2
        'n_heads': 1,           # Reduced from 2
        'p': 0.5,
        'act': nn.GELU(),
        'lr': 1e-4
    }

    config.learner = {
        'cls': 'GradientAscentDiscrete',
        'keep': True,
        'design_steps': 50,
        'decay': 0.,
        'lr': 2
    }

    return config
