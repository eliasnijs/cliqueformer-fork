import numpy as np 
import torch 
import os 

from copy import deepcopy
from data.dataset import Dataset
from data.extras import get_data_below_percentile


class TFBind8(Dataset):

    def __init__(self, **kwargs):
        
        data_path = os.path.join("scrape", "TF-Bind")
        x_path = os.path.join(data_path, "tf_bind_8-x-0.npy")
        y_path = os.path.join(data_path, "tf_bind_8-y-0.npy")

        X = np.load(x_path)
        Y = np.load(y_path)

        self.seq_len = X.shape[-1]
        self.x_all = X
        self.y_all = Y
        self.lookup = {
            tuple(X[i]): Y[i] for i in range(len(X))
        }

        X = np.eye(4)[X]
        x, y, _ = get_data_below_percentile(X, Y, 80)
        
        super().__init__(x, y, **kwargs)

    def evaluate(self, x, from_standardized_x=False, to_standardized_y=False):

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = x.argmax(-1)
        
        y = np.array([
            self.lookup[tuple(x[i])] for i in range(len(x))
        ])

        if to_standardized_y:
            return self.to_standard_y(y)

        return y 
        
