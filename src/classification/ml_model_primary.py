"""
Apply XGBoost to the primary dataset.
"""

# Import dependencies
import os
import pandas as pd
import numpy as np
import random
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Set relative path
my_path = os.getcwd()

# Set the seed
default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)

# Import Reddit features and labels for error analysis
truth_x_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_x_train.csv')
truth_y_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_y_train.csv', header=None)

# Import primary data
primary_df = pd.read_csv(my_path + '/data/cleaned/with_features/df_primary.csv')

#region DATA PREPARATION

# Transform data into matrices and vectors
truth_x_train1 = truth_x_train.drop(['Unnamed: 0', 'index'], axis=1)
truth_x_train1 = truth_x_train1.values
truth_y_train1 = truth_y_train.iloc[1:, 1].values
truth_y_train1 = truth_y_train1.astype(int)

# Instantiate the standard scaler
sc = StandardScaler()

# Standardize the feature matrix
truth_x_train1 = sc.fit_transform(truth_x_train1)

# Prepare the primary dataset by selecting just the features
primary_df1 = primary_df.iloc[:, 4:].values

# Standardize the primary dataset features
sc1 = StandardScaler()
primary_df1 = sc1.fit_transform(primary_df1)

#endregion

#region CLASSIFICATION OF PRIMARY DATASET

# Trouble loading XGBoost model with joblib, so retrain with best hyperparameters
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.17413992398042408,
                    reg_lambda=3, max_depth=33, subsample=0.5578680145837858)

# Fit the XGBoost classifier to the data
xgb.fit(truth_x_train1, truth_y_train1)

# Make predictions
primary_preds = xgb.predict(primary_df1)

# Count predictions
print(primary_preds[primary_preds == 1].shape)
print(primary_preds[primary_preds == 0].shape)

#endregion
