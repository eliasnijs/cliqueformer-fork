import numpy as np 
import torch 
import os 

from xgboost import XGBRegressor
from data.dataset import Dataset

import scrape.autofocused_oracles.util as util 


class Superconductor(Dataset):

    def __init__(self, **kwargs):

        seed = kwargs.pop("seed")
        path = os.path.join("scrape", "autofocused_oracles", "preprocessed_data.npz")
        data = np.load(path)
        x = data["X_nxm"]

        xgb_path = os.path.join("scrape", "autofocused_oracles", "gt_all_feats.model")
        xgb = XGBRegressor(**util.XGB_PARAMS)
        xgb.load_model(xgb_path)
        
        y = xgb.predict(x)
        x, y, _ = util.get_data_below_percentile(x, y, 80, seed=seed)
        self.oracle = xgb
    
        super().__init__(x, y, **kwargs)

    def evaluate(self, x, from_standardized_x = True, to_standardized_y = True):

        if isinstance(x, torch.Tensor):
           x = x.detach().cpu().numpy()

        if from_standardized_x:
            x = self.to_raw_x(x)

        y = self.oracle.predict(x)
        
        if to_standardized_y:
            return self.to_standard_y(y)
        
        return y 