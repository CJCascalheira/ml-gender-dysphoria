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
truth_depression = pd.read_csv(my_path + '\data\cleaned\dass\\truth\\truth_depression.csv')
truth_anxiety = pd.read_csv(my_path + '\data\cleaned\dass\\truth\\truth_anxiety.csv')
truth_stress = pd.read_csv(my_path + '\data\cleaned\dass\\truth\\truth_stress.csv')
truth_suicide = pd.read_csv(my_path + '\data\cleaned\dass\\truth\\truth_suicide.csv')

# Load the primary datasets
primary_depression = pd.read_csv(my_path + '\data\cleaned\dass\primary\primary_depression.csv')
primary_anxiety = pd.read_csv(my_path + '\data\cleaned\dass\primary\primary_anxiety.csv')
primary_stress = pd.read_csv(my_path + '\data\cleaned\dass\primary\primary_stress.csv')
primary_suicide = pd.read_csv(my_path + '\data\cleaned\dass\primary\primary_suicide.csv')

# Load the SVM models
svm_depression = load(my_path + '\models\dass_depression.joblib')
svm_anxiety = load(my_path + '\models\dass_anxiety.joblib')
svm_stress = load(my_path + '\models\dass_stress.joblib')
svm_suicide = load(my_path + '\models\dass_suicide.joblib')

# endregion

# region ENGINEER DASS FEATURES FROM SVM-GENERATED LABELS

# Predict depression

# Predict anxiety

# Predict stress

# Predict suicide

# endregion
