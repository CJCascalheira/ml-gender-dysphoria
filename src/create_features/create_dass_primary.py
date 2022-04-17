"""
Create DASS features for both the primary dataset.
"""

# region PREPARE WORKSPACE

# Import dependencies
import os
import pandas as pd
from joblib import load

# Get current working directory
my_path = os.getcwd()

# Load the primary datasets
primary_depression = pd.read_csv(my_path + '/data/cleaned/dass/primary/primary_depression.csv')
primary_anxiety = pd.read_csv(my_path + '/data/cleaned/dass/primary/primary_anxiety.csv')
primary_stress = pd.read_csv(my_path + '/data/cleaned/dass/primary/primary_stress.csv')
primary_suicide = pd.read_csv(my_path + '/data/cleaned/dass/primary/primary_suicide.csv')

# Load the SVM models
svm_depression = load(my_path + '/models/dass_depression.joblib')
svm_anxiety = load(my_path + '/models/dass_anxiety.joblib')
svm_stress = load(my_path + '/models/dass_stress.joblib')
svm_suicide = load(my_path + '/models/dass_suicide.joblib')

# endregion

# region PREPARE DATA

# Get features
x_depression = primary_depression.drop(['text', 'dysphoria'], axis=1)
x_anxiety = primary_anxiety.drop(['text', 'dysphoria'], axis=1)
x_stress = primary_stress.drop(['text', 'dysphoria'], axis=1)
x_suicide = primary_suicide.drop(['text', 'dysphoria'], axis=1)

# endregion

# region ENGINEER DASS FEATURES FROM SVM-GENERATED LABELS

# Start file output
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('##############################################################', file=f)
    print('SVM CLASSIFICATION OF DASS - PRIMARY #########################', file=f)
    print('\n', file=f)

# Predict depression
y_depression = svm_depression.predict(x_depression)
pd.DataFrame(y_depression).to_csv(my_path + '/data/cleaned/dass/primary/features/y_depression.csv')

# Update output file
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('SVM classification of depression: FINISHED!', file=f)
    print('\n', file=f)

# Predict anxiety
y_anxiety = svm_anxiety.predict(x_anxiety)
pd.DataFrame(y_anxiety).to_csv(my_path + '/data/cleaned/dass/primary/features/y_anxiety.csv')

# Update output file
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('SVM classification of anxiety: FINISHED!', file=f)
    print('\n', file=f)

# Predict stress
y_stress = svm_stress.predict(x_stress)
pd.DataFrame(y_stress).to_csv(my_path + '/data/cleaned/dass/primary/features/y_stress.csv')

# Update output file
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('SVM classification of stress: FINISHED!', file=f)
    print('\n', file=f)

# Predict suicide
y_suicide = svm_suicide.predict(x_suicide)
pd.DataFrame(y_suicide).to_csv(my_path + '/data/cleaned/dass/primary/features/y_suicide.csv')

# Update output file
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('SVM classification of suicide: FINISHED!', file=f)
    print('\n', file=f)

# endregion
