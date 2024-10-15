import numpy as np 
import torch 
import math 
from scipy.special import softmax

from data.dataset import Dataset



def rbf_mixture(x, cliques, means, weights):

  #
  # Initialize the vector of evaluations of the RBF
  #
  y = np.zeros((x.shape[0],))

  for i, clique in enumerate(cliques):
    #
    # Extract the variables of a given clique
    #
    x_clique = x[:, clique]

    #
    # Add the corresponding sub-function to the total value
    #
    y += (
        weights[i] * np.exp(-np.linalg.norm(x_clique - means[i], axis=-1)**2)
    )

  return y


def softplus(x):
   return np.abs(x) + np.log( np.exp(-np.abs(x)) + np.exp(x - np.abs(x)) )


def inverse_softplus(x):
   return np.abs(x) + np.log( np.exp(x - np.abs(x)) - np.exp(-np.abs(x)))



class LRBF(Dataset):

    def __init__(self, N, d, **kwargs):

        if d % 2 == 0:
           d += 1 
        
        #
        # Build cliques for this dataset
        #
        cliques = [
           [2*i, 2*i+1, 2*i+2] for i in range((d-1)//2)
        ]
        n_cliques = len(cliques)

        #
        # Initialize the random parameters of the RBF
        #
        means = [
            np.random.randn(len(clique)) for clique in cliques
        ]
        weights = softmax(
            np.random.randn(n_cliques)/math.sqrt(n_cliques)
        )

        #
        # Make the function and the targets
        #
        self.f = (lambda x: rbf_mixture(x, cliques, means, weights))
        z = np.random.randn(N, d)
        y = self.f(z)

        #
        # Create a non-linear transformation of the data
        #
        d_obs = d + 10
        self.transform_lin = np.random.randn(d, d_obs)/math.sqrt(d)
        self.transform_bias = np.random.randn(1, d_obs)


        self.transform = (lambda x: softplus(x @ self.transform_lin + self.transform_bias))
        self.inverse_transform = (lambda x: (inverse_softplus(x) - self.transform_bias)@np.linalg.pinv(self.transform_lin))
        x = self.transform(z)

        #
        # Inherit from Dataset
        #
        super().__init__(x, y, **kwargs)

    def evaluate(self, x, from_standardized_x = True, to_standardized_y = True):
        
        if isinstance(x, torch.Tensor):
           x = x.detach().cpu().numpy()

        if from_standardized_x:
            x = self.to_raw_x(x)

        z = self.inverse_transform(x)
        y = self.f(z)

        if to_standardized_y:
            return self.to_standard_y(y)
        return y 