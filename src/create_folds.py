import numpy as np
import pandas as pd

import config

from sklearn import model_selection

if __name__ == "__main__":
   df = pd.read_csv(config.TRAINING_PATH)
   df['kfold'] = -1
   df = df.sample(frac=1).reset_index(drop=True)

   kf = model_selection.KFold(n_splits=5)

   for fold_ ,(t_, v_) in enumerate(kf.split(X=df)):
      df.loc[v_, 'kfold'] = fold_

   df.to_csv(config.FOLDED_TRAINING_PATH, index=False)
