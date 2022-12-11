"""
SVM classifier for the DASS labels.

Trains a depression classifier.
"""

# region PREPARE WORKSPACE

# Load dependencies
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from datetime import datetime
from sklearn import metrics
from joblib import dump

# Get current working directory
my_path = os.getcwd()

# Start file output
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('\n', file=f)
    print('##############################################################', file=f)
    print('DEPRESSION OUTPUT ############################################', file=f)
    print('\n', file=f)

# endregion

# region PREPARE DATA

# Load the data
raw_data = pd.read_csv(my_path + '/data/cleaned/dass/with_features/depression.csv')

# Get features and label
X = raw_data.drop(['id', 'text', 'label', 'dysphoria'], axis=1)
Y = raw_data['label']

# No standardization needed because features are dichotomous

# Split into 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1, stratify=Y)

# endregion

# region SVM CLASSIFIER

# Instantiate the class
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Prepare the K-fold
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Set the metrics
my_metrics = ['accuracy', 'precision', 'recall', 'f1']

# Perform k-fold cross-validation
scores = cross_validate(estimator=svm, X=x_train, y=y_train, scoring=my_metrics, cv=kfold, n_jobs=-1,
                        error_score='raise')

# Print the average scores during training
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('TRAINING METRICS', file=f)
    print('Average runtime: %.3f' % np.mean(scores['fit_time'] + scores['score_time']), file=f)
    print('Average accuracy: %.3f (%.3f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])),
          file=f)
    print('Average precision: %.3f (%.3f)' % (np.mean(scores['test_precision']), np.std(scores['test_precision'])),
          file=f)
    print('Average recall: %.3f (%.3f)' % (np.mean(scores['test_recall']), np.std(scores['test_recall'])), file=f)
    print('Average F1: %.3f (%.3f)' % (np.mean(scores['test_f1']), np.std(scores['test_f1'])), file=f)
    print('\n', file=f)

# Fit the data
start_time = datetime.now()
svm.fit(x_train, y_train)
end_time = datetime.now()

# Save results to file
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('Runtime to fit SVM model: ' + str(end_time - start_time), file=f)
    print('\n', file=f)

# Get the predicted class labels
start_time = datetime.now()
y_pred = svm.predict(x_test)
end_time = datetime.now()

# Save results to file
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('Runtime to predict class labels: ' + str(end_time - start_time), file=f)
    print('\n', file=f)

# Print the metrics of the test results
with open(my_path + '/doc/dass_output.txt', 'a') as f:
    print('TEST METRICS', file=f)
    print('Accuracy: %.3f' % metrics.accuracy_score(y_true=y_test, y_pred=y_pred), file=f)
    print('Precision: %.3f' % metrics.precision_score(y_true=y_test, y_pred=y_pred), file=f)
    print('Recall: %.3f' % metrics.recall_score(y_true=y_test, y_pred=y_pred), file=f)
    print('F1: %.3f' % metrics.f1_score(y_true=y_test, y_pred=y_pred), file=f)
    print('\n', file=f)

# endregion

# Save the SVM model to a file
dump(svm, my_path + '/models/dass_depression.joblib')
