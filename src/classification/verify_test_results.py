"""
Execute XGBoost again with a new permutation of data to verify the test results.
"""

#region IMPORT DATA AND LOAD LIBRARIES

# Load dependencies
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Set working directory
my_path = os.getcwd()

# Set the seed
random.seed(1234567)

#endregion

#region PREPARE DATA AND METRICS

# Import the data
df_truth = pd.read_csv(my_path + '/data/cleaned/with_features/df_truth.csv')

# Get the features and the labels
truth_x = df_truth.drop(['Unnamed: 0', 'temp_id', 'text', 'dysphoria'], axis=1)
truth_y = df_truth['dysphoria']

# Split the data into training and test sets
truth_x_train, truth_x_test, truth_y_train, truth_y_test = train_test_split(truth_x, truth_y, test_size=0.20, stratify=truth_y)

# Number of examples in each set
print(truth_x_train.shape)
print(truth_x_test.shape)

# Transform to matrices
truth_x_train = truth_x_train.values
truth_x_test = truth_x_test.values
truth_y_train = truth_y_train.values
truth_y_test = truth_y_test.values

# Instantiate the standard scaler
sc = StandardScaler()

# Standardize the feature matrix
truth_x_train = sc.fit_transform(truth_x_train)
truth_x_test = sc.transform(truth_x_test)

#endregion

#region EXECUTE XGBOOST

# Trouble loading XGBoost model with joblib, so retrain with best hyperparameters
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.17413992398042408,
                    reg_lambda=3, max_depth=33, subsample=0.5578680145837858)

# Fit the XGBoost classifier to the data
xgb.fit(truth_x_train, truth_y_train)

# Predict the outcomes in the test set
y_pred = xgb.predict(truth_x_test)

# Print metrics
print("Accuracy is %.3f" % accuracy_score(y_true=truth_y_test, y_pred=y_pred))
print("Precision is %.3f" % precision_score(y_true=truth_y_test, y_pred=y_pred))
print("Recall is %.3f" % recall_score(y_true=truth_y_test, y_pred=y_pred))
print("F1 score is %.3f" % f1_score(y_true=truth_y_test, y_pred=y_pred))
print("ROC/AUC score is %.3f" % roc_auc_score(y_true=truth_y_test, y_score=y_pred))

#endregion