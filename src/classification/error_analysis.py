"""
Perform error analysis on the ground truth dataset.

RESOURCES:
- sklearn documentation
- https://github.com/dataiku-research/mealy
- https://dataiku-research.github.io/mealy/reference.html
- https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
"""

# Import dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from mealy.error_analyzer import ErrorAnalyzer
from mealy.error_visualizer import ErrorVisualizer

# Set relative path
my_path = os.getcwd()

# Set the backend for matplotlib
matplotlib.use('Qt5Agg')

# Set the seed
default_seed = 10
np.random.seed(default_seed)
random.seed(default_seed)

# Import predictions and true labels
predictions_xgb_test = pd.read_csv(my_path + '/data/results/confusion_matrix_data/predictions_xgb_test.csv')
predictions_xgb_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/predictions_xgb_train.csv')

# Import Reddit features and labels for error analysis
truth_x_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_x_train.csv')
truth_x_test = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_x_test.csv')
truth_y_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_y_train.csv', header=None)
truth_y_test = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_y_test.csv', header=None)

# Import Reddit text for error analysis
df_truth = pd.read_csv(my_path + '/data/cleaned/with_features/df_truth.csv')

#region CONFUSION MATRIX

# Add false positive column
predictions_xgb_test['false_pos'] = np.where((predictions_xgb_test.loc[:, 'truth_y_test'] == 1) &
                                         (predictions_xgb_test.loc[:, 'y_pred_test'] == 0), 1, 0)

predictions_xgb_train['false_pos'] = np.where((predictions_xgb_train.loc[:, 'truth_y_train'] == 1) &
                                         (predictions_xgb_train.loc[:, 'y_pred_train'] == 0), 1, 0)
# Add false negative column
predictions_xgb_test['false_neg'] = np.where((predictions_xgb_test.loc[:, 'truth_y_test'] == 0) &
                                         (predictions_xgb_test.loc[:, 'y_pred_test'] == 1), 1, 0)

predictions_xgb_train['false_neg'] = np.where((predictions_xgb_train.loc[:, 'truth_y_train'] == 0) &
                                         (predictions_xgb_train.loc[:, 'y_pred_train'] == 1), 1, 0)

# Set visualization preferences
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 30})

# Visualize the confusion matrix of test set
ConfusionMatrixDisplay.from_predictions(y_true=predictions_xgb_test['truth_y_test'], y_pred=predictions_xgb_test['y_pred_test'])

# Visualize the confusion matrix of the train set
ConfusionMatrixDisplay.from_predictions(y_true=predictions_xgb_train['truth_y_train'], y_pred=predictions_xgb_train['y_pred_train'])

#endregion

#region MEALY ANALYSIS

# Prepare data for error tree model
truth_x_train1 = truth_x_train.drop(['Unnamed: 0', 'index'], axis=1)
truth_x_test1 = truth_x_test.drop(['Unnamed: 0', 'index'], axis=1)

# Get the feature names
feature_names = truth_x_train1.columns

# Transform data into matrices and vectors
truth_x_train1 = truth_x_train1.values
truth_x_test1 = truth_x_test1.values
truth_y_train1 = truth_y_train[1].values
truth_y_test1 = truth_y_test[1].values

# Instantiate the standard scaler
sc = StandardScaler()

# Standardize the feature matrix
truth_x_train1 = sc.fit_transform(truth_x_train1)
truth_x_test1 = sc.transform((truth_x_test1))

# Trouble loading XGBoost model with joblib, so retrain with best hyperparameters
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.17413992398042408,
                    reg_lambda=3, max_depth=33, subsample=0.5578680145837858)

# Fit the XGBoost classifier to the data
xgb.fit(truth_x_train1, truth_y_train1)

# Instantiate and train the error analyzer
error_analyzer = ErrorAnalyzer(xgb, feature_names=feature_names)
error_analyzer.fit(truth_x_test1, truth_y_test1)

# Print error tree metrics
print(error_analyzer.evaluate(truth_x_test1, truth_y_test1, output_format='str'))

# Print the details regarding the decision tree nodes containing the majority of errors
pprint(error_analyzer.get_error_leaf_summary(leaf_selector=None, add_path_to_leaves=True))

# Instantiate the error visualizer
error_visualizer = ErrorVisualizer(error_analyzer)

# Set the leaf id to correspond to the lead node with the most error
leaf_id = error_analyzer._get_ranked_leaf_ids()[0]

# Visualize the error
error_visualizer.plot_feature_distributions_on_leaves(leaf_selector=leaf_id, top_k_features=10)

#endregion

#region IDENTIFY FALSE POSITIVES AND NEGATIVES TEXT

# Keep the text from the ground truth data set
df_truth = df_truth.rename(columns={'Unnamed: 0': 'index'})
df_truth = df_truth.loc[:, ('index', 'text')]

# Select the ground truth examples in the test set
df_truth_test = df_truth[df_truth['index'].isin(truth_x_test['index'].tolist())]
df_truth_test = df_truth_test.reset_index()

# Concatenate test set
df_test = pd.concat([df_truth_test, predictions_xgb_test], axis=1)

# Select the columns to keep
df_test1 = df_test.loc[:, ('index', 'text', 'false_pos', 'false_neg')]

# Keep examples of false neg or false pos
df_test2 = df_test1[(df_test1['false_neg'] == 1) | (df_test1['false_pos'] == 1)]

# Export false neg / pos for illustrative examples
df_test2.to_csv(my_path + '/data/results/xgb_misclassification_examples.csv')

#endregion
