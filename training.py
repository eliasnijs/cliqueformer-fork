import numpy as np
import torch
import torch.nn as nn

import wandb
import pickle 
import os 

from absl import app, flags
from ml_collections import config_flags

from models import Cliqueformer, CliqueformerDiscrete, COMs, COMsDiscrete, Naive, NaiveDiscrete, MBOTransformer, MBOTransformerDiscrete
from optimization.design import Design
from optimization.lerners import GradientAscent, GradientAscentDiscrete, RWR, RWRDiscrete 
from data import Dataset, LRBF, Superconductor, TFBind8, DNA
import data.extras as dx 
import models.graphops as graphops

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', int(1), 'Random seed.') 
flags.DEFINE_integer('batch_size', int(512), 'Batch size.') 
flags.DEFINE_integer('design_batch_size', int(1000), 'Design batch size.') 
flags.DEFINE_integer('model_steps', int(4e4), 'Model learning size.') 
flags.DEFINE_integer('beta_warmup', int(1e3), 'Number of KL warmup steps.')
flags.DEFINE_integer('N_eval', int(2e2), 'Evaluation frequency.')
flags.DEFINE_integer('top_k', int(10), 'The best designs for evaluation.')
flags.DEFINE_float('split_ratio', 0.8, 'Train-test split.') 

config_flags.DEFINE_config_file(
    'config',
    'configs/superconductor/cliqueformer.py',
    'File with hyperparameter configurations.',
    lock_config=False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(_):

    #
    # Initialize a Wandb project
    #
    wandb.init(project='cliqueformer-training')
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
    # Build model spec tring for saving
    #
    spec = ', '.join([f'{key}: {value}' for key, value in model_kwargs.items()])

    #
    # Initialize the dataset
    #
    data_kwargs["seed"] = FLAGS.seed
    dataset_cls = data_kwargs.pop('cls')
    dataset = globals()[dataset_cls](**data_kwargs)

    if dataset_cls not in ["DNA", "TFBind8"]:
        dataset.standardize_x()
    
    dataset.standardize_y()

    #
    # Split the dataset into train-test partition
    #
    dataset_train, dataset_test = dataset.split(FLAGS.split_ratio)

    #
    # Initialize the model
    #
    model_cls = model_kwargs.pop('cls')
    
    if "Discrete" in model_cls:
        model = globals()[model_cls](dataset.seq_len, dataset.dim, **model_kwargs).to(device) 

    else:
        model = globals()[model_cls](dataset.dim, **model_kwargs).to(device) 
    
    model = nn.DataParallel(model)

    #
    # Train the model 
    #
    model.train()

    for step in range(FLAGS.model_steps):
        #
        # Draw a random batch and put it on torch device
        #   
        x, y = dataset_train.sample(FLAGS.batch_size)
        x, y = dx.move_to_device((x, y), device)

        #
        # Compute the loss and take a gradient step
        #
        info = model.module.training_step(x, y)

        if "Cliqueformer" in model_cls:
            model.module.beta_vae = min(1, step/FLAGS.beta_warmup)

        #
        # Evaluate the model on the test set
        #
        if step % FLAGS.N_eval == 0:

            model.eval()

            x, y = dataset_test.sample(FLAGS.batch_size)
            x, y = dx.move_to_device((x, y), device)

            evalinfo = model.module.eval_step(x, y)

            wandb.log({f'eval/{k}': v for k, v in evalinfo.items()}, step=step)
            wandb.log({f'train/{k}': v for k, v in info.items()}, step=step)
            model.train()

    #
    # Save the model
    #
    model_dir = os.path.join('saved_models', model_cls, dataset_cls)
    if dataset_cls == 'LRBF':
        model_dir += str(data_kwargs['d'])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, spec) + '.pickle'
    with open(model_path, 'wb') as model_file:
        pickle.dump(model.module, model_file)

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
            'valid': valid.mean()
        }
        wandb.log({f'ascend/{k}': v for k, v in ascendinfo.items()}, step=FLAGS.model_steps + step)



if __name__ == '__main__':
    app.run(main)