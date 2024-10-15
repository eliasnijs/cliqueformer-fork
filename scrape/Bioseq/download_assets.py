import numpy as np 
import pandas as pd 
import sys 

from src.utils.sequence import seqs_to_one_hot

import wandb

run = wandb.init(project="Diffusion-DNA-RNA")
artifact = run.use_artifact('fderc_diffusion/Diffusion-DNA-RNA/DNA-dataset:v0')
dir = artifact.download()
wandb.finish()

# Check Data 
datafile = pd.read_csv("scrape/Bioseq/artifacts/DNA-dataset:v0/dataset.csv.gz")
datafile.head()

small_data = datafile.loc[ (datafile['chrom'] =="chr1") | (datafile['chrom'] =="chr2") |(datafile['chrom'] =="chr3") |(datafile['chrom'] =="chr4")  ]

seq_x = [seqs_to_one_hot(seq)[:, 0, :] for seq in small_data['seq']]
x = np.array(seq_x)
print(x.shape) # Data Size 