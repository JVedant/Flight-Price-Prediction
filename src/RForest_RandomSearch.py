import pandas as pd
import numpy as np

import config

from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics

if __name__ == "__main__":

   data = pd.read_csv(config.FOLDED_TRAINING_PATH)

   X = data.drop(['Price', 'kfold'], 1).values
   y = data.Price.values

   forest = ensemble.RandomForestRegressor(n_jobs=-1)

   params = {
      "n_estimators": np.arange(100, 1500, 100),
      "criterion" : ["mse", "mae"],
      "max_depth": np.arange(1, 31)
   }

   model = model_selection.RandomizedSearchCV(
      estimator=forest,
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
didn't get the params as the model took 20 mins to train for 1 iter - 1 fold,
if you got a high processing CPU, feel free to check it

"""