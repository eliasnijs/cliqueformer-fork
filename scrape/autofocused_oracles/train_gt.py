import numpy as np
from xgboost import XGBRegressor  # using xgboost version 0.90
from sklearn.feature_selection import SelectFromModel
import util as util

import matplotlib.pyplot as plt
import seaborn as sns


d = np.load("preprocessed_data.npz")
X_nxm = d["X_nxm"]  # zero-mean unit-variance features from Hamidieh (2018) dataset
labels_n = d["y_n"]  # original labels from Hamidieh (2018) dataset
print("X_nxm shape: {}".format(X_nxm.shape))
print("labels_n shape: {}".format(labels_n.shape))

gtfs = XGBRegressor(**util.XGB_PARAMS)
n_train = int(0.8 * labels_n.size)
Xtr_nxm, Xval_nxm = X_nxm[: n_train, :], X_nxm[n_train :, :]
labelstr_n, labelsval_n = labels_n[: n_train], labels_n[n_train :]
gtfs.fit(Xtr_nxm, labelstr_n,
         eval_set=[(Xtr_nxm, labelstr_n), (Xval_nxm, labelsval_n)],
        eval_metric="rmse", early_stopping_rounds=20, verbose=2)

gtfs.save_model("gt_all_feats.model")


gt = XGBRegressor(**util.XGB_PARAMS)
gt.load_model("gt_all_feats.model")
y = gt.predict(X_nxm)

X, Y, _ = util.get_data_below_percentile(X_nxm, y, 80)