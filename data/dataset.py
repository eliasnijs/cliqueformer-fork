import torch 
import numpy as np 


class Dataset:

    def __init__(self, x, y, **kwargs): 
        self.x = x
        self.dim = x.shape[-1]
        self.y = y 
        self.x_standardized = False 
        self.y_standardized = False
        self.x_mean = None 
        self.x_std = None 
        self.y_mean = None 
        self.y_std = None  
        self.y_min = y.min()
        self.y_max = y.max()

    def standardize_x(self):

        if self.x_standardized:
            return 
        
        self.x_mean = self.x.mean(0)
        self.x_std = self.x.std(0)
        
        self.x = (self.x - self.x_mean) / (self.x_std + 1e-8)
        self.x_standardized = True 
    
    def unstandardize_x(self):

        if not self.x_standardized:
            return 

        self.x = self.x * (self.x_std + 1e-8) + self.x_mean
        self.x_standardized = False 

    def to_raw_x(self, x):
        return x * (self.x_std + 1e-8) + self.x_mean
    
    def standardize_y(self):
        
        if self.y_standardized:
            return 
        
        self.y_mean = self.y.mean(0)
        self.y_std = self.y.std(0)

        self.y = (self.y - self.y_mean) / (self.y_std + 1e-8)
        self.y_standardized = True

    def to_standard_y(self, y):
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def unstandardize_y(self):

        if not self.y_standardized:
            return 
        
        self.y = self.y * (self.y_std + 1e-8) + self.y_mean
        self.y_standardized = False

    def max_min_normalize(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)

    def sample(self, n):
        indexes = np.random.randint(0, self.size, n)
        x, y = self.x[indexes], self.y[indexes]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y


    @property
    def data(self):
        return (self.x, self.y)

    @property
    def size(self):
        return self.x.shape[0]

    def split(self, ratio=0.8):

        N = self.size 
        N_train = int(ratio * N)

        ix = np.random.permutation(N)

        data_train = Dataset(self.x[ix[:N_train]], self.y[ix[:N_train]])
        data_test = Dataset(self.x[ix[N_train:]], self.y[ix[N_train:]])

        return data_train, data_test
    

