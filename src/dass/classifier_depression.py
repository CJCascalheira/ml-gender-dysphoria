"""
SVM classifier for the DASS labels.

Trains a depression classifier.
"""

# region PREPARE WORKSPACE

# Load dependencies
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Get current working directory
my_path = os.getcwd()

# Start file output
with open(my_path + '\doc\dass_output.txt', 'a') as f:
    print('##############################################################', file=f)
    print('DEPRESSION OUTPUT ############################################', file=f)
    print('\n', file=f)

# endregion

# region PREPARE DATA

# Load the data
raw_data = pd.read_csv(my_path + '\data\cleaned\dass\with_features\depression.csv')

# DELETE LATER =====================================================================
raw_data = raw_data.sample(n=2000)
# ==================================================================================

# Get features and label
X = raw_data.drop(['id', 'text', 'label'], axis=1)
Y = raw_data['label']

# No standardization needed because features are dichotomous

# Split into 80% train, 20% test
x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.20, random_state=1, stratify=Y)

# endregion