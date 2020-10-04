import pandas as pd
import numpy as np

import config

from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

if __name__ == "__main__":

   data = pd.read_csv(config.FOLDED_TRAINING_PATH)

   X = data.drop(['Price', 'kfold'], 1).values
   y = data.Price.values

   Dtree = tree.DecisionTreeRegressor()

   params = {
      "criterion" : ["mse", "friedman_mse", "mae"],
      "splitter" : ["best", "random"],
      "max_depth": np.arange(1, 31)
   }

   model = model_selection.RandomizedSearchCV(
      estimator=Dtree,
      cv=5,
      verbose=10,
      param_distributions=params,
      n_iter=20,
      n_jobs=1,
   )

   model.fit(X, y)
   print(f"best score = {model.best_score_}")

   print(f"best params = {model.best_params_}")

"""
mae= 4167.134636090804
mae= 4084.9948456517945
mae= 4029.8228121827765
mae= 4044.6031214001014
mae= 4079.9367955233142
"""