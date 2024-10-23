import numpy as np 
import pandas as pd 

from src.utils.sequence import seqs_to_one_hot

import wandb

run = wandb.init(project="Diffusion-DNA-RNA")
artifact = run.use_artifact('fderc_diffusion/Diffusion-DNA-RNA/DNA-dataset:v0')
dir = artifact.download()
wandb.finish()

