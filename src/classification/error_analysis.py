"""
Perform error analysis on the ground truth dataset.

RESOURCES:
- sklearn documentation
- https://github.com/dataiku-research/mealy
- https://dataiku-research.github.io/mealy/reference.html
- https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
-
"""

# Import dependencies
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

# Set relative path
my_path = os.getcwd()

# Set the backend for matplotlib
matplotlib.use('Qt5Agg')

# Import predictions and true labels
predictions_xgb = pd.read_csv(my_path + '/data/results/confusion_matrix_data/predictions_xgb.csv')

#region CONFUSION MATRIX

# Calculate confusion matrix of test set
confusion_matrix(y_true=predictions_xgb['truth_y_test'], y_pred=predictions_xgb['y_pred_test'])

# Visualize the confusion matrix of test set
ConfusionMatrixDisplay.from_predictions(y_true=predictions_xgb['truth_y_test'], y_pred=predictions_xgb['y_pred_test'])
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 30})

# Screenshot the plot to save
plt.show()

#endregion
