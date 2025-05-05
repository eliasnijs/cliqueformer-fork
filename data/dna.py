import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn 
import pickle
from grelu.lightning import LightningModel

# Use consistent device across the codebase - using CPU for now to avoid memory issues
device = torch.device("cpu")

import os 
from data.dataset import Dataset
from data.extras import get_data_below_percentile
from scrape.Bioseq.src.utils.sequence import seqs_to_one_hot


#
# The dataset and the pre-trained reward model comes from repository https://github.com/masa-ue/RLfinetuning_Diffusion_Bioseq.
#

class DNA(Dataset):

    def __init__(self, dna_property="hepg2", **kwargs):
        
        self.dna_property = dna_property
        properties = ["hepg2", "k562", "sknsh"]
        self.target_index = properties.index(self.dna_property)

        #
        # Fetch the model
        #
        model_path = os.path.join("scrape", "Bioseq", "artifacts", "DNA-model:v0", "reward_model.ckpt")
        
        # Explicitly map the model to CPU first
        model = LightningModel.load_from_checkpoint(model_path, map_location="cpu")
        
        # Move model to device
        model = model.to(device)
        
        # Only use DataParallel if there's more than one GPU
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        self.oracle = model
        self.oracle.eval()

        from_existing = kwargs.pop("from_existing")
        ready_data_path = os.path.join("scrape", "Bioseq", "artifacts", "DNA-dataset:v0", "ready_data.pickle")

        #
        # Load ready to use data from pickle
        #
        if from_existing:
            with open(ready_data_path, 'rb') as file:
                loaded_data = pickle.load(file)
            x, y = loaded_data
        #
        # Make rady data from raw data
        #
        else:
            #
            # Get the raw data
            #
            data_path = os.path.join("scrape", "Bioseq", "artifacts", "DNA-dataset:v0", "dataset.csv.gz")
            df = pd.read_csv(data_path)

            #
            # Extract the designs 
            # 
            df_subset = df.loc[ (df['chrom'] =="chr1") | (df['chrom'] =="chr2") |(df['chrom'] =="chr3") | (df['chrom'] =="chr4")  ]
            seq_x = [seqs_to_one_hot(seq)[:, 0, :] for seq in df_subset['seq']]
            x = np.array(seq_x)
            n_data = x.shape[0]

            #
            # Prepare for oracle predictions
            #
            y = np.zeros((x.shape[0], 3))
            batch_size = 128  # Reduced batch size to save memory
            start = 0

            while start < n_data:

                batch_x = x[start : start + batch_size]
                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_x = torch.permute(batch_x, (0, 2, 1))

                pred = model(batch_x).squeeze(-1)
                y[start : start + batch_size] = pred.detach().cpu().numpy()
                start += batch_size
        
            #
            # Save in pickle
            #
            with open(ready_data_path, 'wb') as file:
                pickle.dump((x, y), file)

        #
        # Drop redundant data
        #
        y = y[..., self.target_index]            
        x, y, _ = get_data_below_percentile(x, y)
        self.seq_len = x.shape[-2]
        
        super().__init__(x, y, **kwargs)

    def evaluate(self, x, from_standardized_x=False, to_standardized_y=False):

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        
        with torch.no_grad():
            x = torch.permute(x, (0, 2, 1))
            y = self.oracle(x).squeeze(-1)[..., self.target_index]
        
        y = y.cpu().numpy()

        if to_standardized_y:
            return self.to_standard_y(y)
        
        return y 
