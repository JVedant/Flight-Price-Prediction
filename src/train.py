import pandas as pd
import numpy as np

import config
import joblib
import os

from sklearn import ensemble
from sklearn import metrics

if __name__ == "__main__":
   df = pd.read_csv(config.FOLDED_TRAINING_PATH)

   X = df.drop(['Price', 'kfold'], 1).values
   y = df.Price.values

   model = ensemble.RandomForestRegressor()
   model.fit(X, y)

   joblib.dump(
         model,
         os.path.join(config.MODEL_PATH, "model.pkl")
      )
