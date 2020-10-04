import numpy as np
import pandas as pd

import config

from sklearn import metrics
from sklearn import linear_model

def run(fold):
   df = pd.read_csv(config.FOLDED_TRAINING_PATH)

   df_train = df[df.kfold != fold].reset_index(drop=True)
   df_valid = df[df.kfold == fold].reset_index(drop=True)

   x_train = df_train.drop(['kfold', 'Price'], 1).values
   y_train = df_train.Price.values

   x_valid = df_valid.drop(['kfold', 'Price'], 1).values
   y_valid = df_valid.Price.values

   model = linear_model.LinearRegression()

   model.fit(x_train, y_train)
   predict = model.predict(x_valid)

   mae = metrics.mean_absolute_error(y_valid, predict)

   print(f"Fold= {fold}, mae= {mae}")

if __name__ == "__main__":
   for fold_ in range(5):
      run(fold_)


"""
Fold= 0, mae= 4329.861126100114
Fold= 1, mae= 4266.434484772078
Fold= 2, mae= 4193.900757473028
Fold= 3, mae= 4111.301653763441
Fold= 4, mae= 4240.335312225889
"""