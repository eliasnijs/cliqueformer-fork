import numpy as np
import torch
import torch.nn as nn
import argparse

import wandb
import pickle 
import os 

from absl import app, flags
from ml_collections import config_flags

from models import Cliqueformer, CliqueformerDiscrete, COMsDiscrete, Naive, NaiveDiscrete
from optimization.design import Design
from optimization.lerners import GradientAscent, RWR, GradientAscentDiscrete, RWRDiscrete 
from data import Dataset, LRBF, Superconductor, DNA, TFBind8
import models.graphops as graphops

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', int(1), 'Random seed.') 
flags.DEFINE_integer('design_batch_size', int(1000), 'Design batch size.') 
flags.DEFINE_integer('top_k', int(10), 'The best designs for evaluation.') 
flags.DEFINE_float('split_ratio', 0.8, 'Train-test split.') 

config_flags.DEFINE_config_file(
    'config',
    'configs/lrbf/cliqueformer.py',
    'File with hyperparameter configurations.',
    lock_config=False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(_):

    #
    # Parser to extract seed
    # 
    parser = argparse.ArgumentParser(description="Pass in the random seed.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    args = parser.parse_args()
    seed = FLAGS.seed if args.seed is None else args.seed

    #
    # Initialize a Wandb project
    #
    wandb.init(project='cliqueformer-optimization')
    wandb.config.update(FLAGS)
    kwargs = dict(**FLAGS.config)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    #
    # Extract specific kwargs
    #
    data_kwargs = dict(kwargs['data'])
    model_kwargs = dict(kwargs['model'])
    learner_kwargs = dict(kwargs['learner'])

    #
    # Build model spec tring for loading
    #
    spec = ', '.join([f'{key}: {value}' for key, value in model_kwargs.items()])

    #
    # Initialize the dataset
    #
    data_kwargs["seed"] = FLAGS.seed
    dataset_cls = data_kwargs.pop('cls')
    dataset = globals()[dataset_cls](**data_kwargs)

    #
    # Change the seed for this run
    #
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_cls not in ["DNA", "TFBind8"]:
        dataset.standardize_x()
    
    dataset.standardize_y()

    #
    # Split the dataset into train-test partition
    #
    dataset_train, _  = dataset.split(FLAGS.split_ratio)

    #
    # Derive the model path
    #
    model_cls = model_kwargs['cls']
    model_dir = os.path.join('saved_models', model_cls, dataset_cls)

    if dataset_cls == 'LRBF':
        model_dir += str(data_kwargs['d'])

    model_path = os.path.join(model_dir, spec + '.pickle')

    #
    # Load the model
    #
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    model = nn.DataParallel(model)

    #
    # Initialize the design from the dataset
    #
    x, _ = dataset_train.sample(FLAGS.design_batch_size)
    init_design = x.to(device)

    if 'Cliqueformer' in model_cls:
        index_matrix = model.module.index_matrix.to(device)
        learner_kwargs['structure_fn'] = (lambda x: graphops.separate_latents(x, index_matrix))
        init_design = model.module.encode(init_design, separate=False)

    design = Design(init_design).to(device)

    #
    # Initialize the learner
    #
    learner_cls = learner_kwargs.pop('cls')
    design_steps = learner_kwargs.pop('design_steps')
    learner_model = nn.DataParallel(model.module.target_regressor)
    learner = globals()[learner_cls](design, learner_model, **learner_kwargs)
    
    #
    # Train the learner
    #
    for step in range(design_steps):
        #
        # Make a train step
        #
        train_info = learner.train_step()
        
        #
        # Obtain the (standardized) grount-truth value of the design
        #
        design = learner.design_fn()

        if 'Cliqueformer' in model_cls:
            with torch.no_grad():
                design = graphops.separate_latents(design, index_matrix)
                design = model.module.decode(design)

        design = design.detach().cpu().numpy()
        true_val = dataset.evaluate(design, from_standardized_x=True, to_standardized_y=False)
        true_val = dataset.max_min_normalize(true_val)

        #
        # Count up nans
        #
        isnan = np.isnan(true_val)
        valid = 1 - isnan

        #
        # Remove nans
        #
        true_val = true_val[np.logical_not(np.isnan(true_val))]

        #
        # Get top-k of design values
        #
        true_val = true_val[np.argsort(true_val)[::-1][:FLAGS.top_k]]

        #
        # Estimate the model's perceived value
        # 
        with torch.no_grad():
            val = learner.value()

        ascendinfo = {
            "model val": val.item(),
            "true val": true_val.mean(),
            "true_val_max": true_val.max(),
            "true_val_std": true_val.std(),
            'valid': valid.mean()

        }
        wandb.log({f'ascend/{k}': v for k, v in ascendinfo.items()}, step=step)



if __name__ == '__main__':
    app.run(main)