"""
Create word embedding and DASS features for both the ground truth and primary datasets.
"""

# region PREPARE WORKSPACE

# Import dependencies
import os
import pandas as pd
from joblib import load

# Get current working directory
my_path = os.getcwd()

# Load the ground truth datasets
df_truth = pd.read_csv(my_path + '\data\cleaned\dass\\truth\\truth_depression.csv')

# Load the primary datasets
df_primary = pd.read_csv(my_path + '\data\cleaned\dass\primary\primary_depression.csv')

# Load the SVM models
svm_depression = load(my_path + '\models\dass_depression.joblib')

# endregion

# region ENGINEER DASS FEATURES FROM SVM-GENERATED LABELS

# Predict depression


# endregion
