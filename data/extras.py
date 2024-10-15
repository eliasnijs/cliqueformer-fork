import numpy as np


def move_to_device(data, device):
    new_data = (t.to(device) for t in data)
    return new_data


def get_data_below_percentile(X_nxp: np.array, y_n: np.array, percentile: float = 80, n_sample: int = None, seed: int = None):

    #
    # Find the data below the percentile
    #
    perc = np.percentile(y_n, percentile)
    idx = np.where(y_n <= perc)[0]
    print("Max label in training data: {:.1f}. {}-th percentile label: {:.1f}".format(np.max(y_n), percentile, perc))

    #
    # Subsample if so specified
    #
    if n_sample is not None and n_sample < idx.size:
        np.random.seed(seed)
        idx = np.random.choice(idx, size=n_sample, replace=False)

    return X_nxp[idx], y_n[idx], idx