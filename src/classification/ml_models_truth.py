"""
Executing several ML models to compare performance using the ground truth dataset, then
performing hyperparameter tuning with random search on the best performing models.

RESOURCES
- https://xgboost.readthedocs.io/en/latest/parameter.html#
- https://jamesrledoux.com/code/randomized_parameter_search
- https://stackoverflow.com/questions/65666164/create-dataframe-with-multiple-arrays-by-column
- https://realpython.com/python-pretty-print/
- Scikit Learn documentation
"""

# Load dependencies
import os
import random
import pandas as pd
import numpy as np
from pprint import pprint
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import truncnorm, randint, uniform

# Set working directory
my_path = os.getcwd()

# Set the seed
random.seed(10)

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

# Save for splits for error analysis
truth_x_train.reset_index(inplace=True)
truth_x_train.to_csv(my_path + '/data/results/confusion_matrix_data/truth_x_train.csv')

pd.DataFrame(truth_y_train).reset_index(inplace=True)
truth_y_train.to_csv(my_path + '/data/results/confusion_matrix_data/truth_y_train.csv')

truth_x_test.reset_index(inplace=True)
truth_x_test.to_csv(my_path + '/data/results/confusion_matrix_data/truth_x_test.csv')

pd.DataFrame(truth_y_test).reset_index(inplace=True)
truth_y_test.to_csv(my_path + '/data/results/confusion_matrix_data/truth_y_test.csv')

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

# Initialize k-fold cross-validation
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Specify the metrics to use
my_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

#endregion

#region LOGISTIC REGRESSION

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=100)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=truth_x_train, y=truth_y_train,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_log_reg['test_accuracy']), np.std(scores_log_reg['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_log_reg['test_precision']), np.std(scores_log_reg['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_log_reg['test_recall']), np.std(scores_log_reg['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_log_reg['test_f1']), np.std(scores_log_reg['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_log_reg['test_roc_auc']), np.std(scores_log_reg['test_roc_auc'])))

#endregion

#region SUPPORT VECTOR MACHINE

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=truth_x_train, y=truth_y_train,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

#endregion

#region DECISION TREE

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=truth_x_train, y=truth_y_train,
               scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

#endregion

#region RANDOM FOREST

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=truth_x_train, y=truth_y_train,
               scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_random_forest['test_accuracy']), np.std(scores_random_forest['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_random_forest['test_precision']), np.std(scores_random_forest['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_random_forest['test_recall']), np.std(scores_random_forest['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_random_forest['test_f1']), np.std(scores_random_forest['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_random_forest['test_roc_auc']), np.std(scores_random_forest['test_roc_auc'])))

#endregion

#region NAIVE BAYES

# Specify hyperparameters of the naive bayes
naive_bayes = GaussianNB()

# Fit the random forest with k-fold cross-validation
scores_naive_bayes = cross_validate(estimator=naive_bayes, X=truth_x_train, y=truth_y_train,
               scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_naive_bayes['test_accuracy']), np.std(scores_naive_bayes['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_naive_bayes['test_precision']), np.std(scores_naive_bayes['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_naive_bayes['test_recall']), np.std(scores_naive_bayes['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_naive_bayes['test_f1']), np.std(scores_naive_bayes['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_naive_bayes['test_roc_auc']), np.std(scores_naive_bayes['test_roc_auc'])))

#endregion

#region XGBOOST

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=truth_x_train, y=truth_y_train,
               scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

#endregion

#region HYPERPARAMETER TUNING - RANDOM SEARCH - RANDOM FOREST

# Create the parameter search space
param_space = {
    # Randomly sample estimators
    'n_estimators': randint(100,1000),

    # Randomly sample numbers
    'max_depth': randint(10,100),
    
    # Normally distributed max_features, with mean .50 stddev 0.15, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.50, scale=0.15)
}

# Instantiate the model
ml_model = RandomForestClassifier()

# Create the random search algorithm
random_search_rf = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='accuracy'
)

# Train the random search algorithm
model_rf = random_search_rf.fit(truth_x_train, truth_y_train)

# Save training results to file
with open(my_path + '/doc/random_search_output.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('TRAINING INFORMATION - RANDOM SEARCH - RANDOM FOREST', file=f)
    print('\nBest Parameters', file=f)
    print(model_rf.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_rf.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_rf.best_index_, file=f)
    print('\nAll Parameters', file=f)
    pprint(model_rf.cv_results_, stream=f)
    print('\n', file=f)

# Predict the training data
y_pred_train = model_rf.predict(truth_x_train)

# Make predictions on the test data
y_pred_test = model_rf.predict(truth_x_test)

# Print the metrics of the test results
with open(my_path + '/doc/random_search_output.txt', 'a') as f:
    print('TEST METRICS W/ BEST MODEL PARAMETERS - RANDOM FOREST', file=f)
    print('Accuracy: %.3f' % metrics.accuracy_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('Precision: %.3f' % metrics.precision_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('Recall: %.3f' % metrics.recall_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('F1: %.3f' % metrics.f1_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('ROC AUC: %.3f' % metrics.roc_auc_score(y_true=truth_y_test, y_score=y_pred_test), file=f)
    print('\n', file=f)

# Save all data for confusion matrix
predictions_df_train = pd.DataFrame(data=zip(truth_y_train, y_pred_train),
             columns=['truth_y_train', 'y_pred_train'])

predictions_df_train.to_csv(my_path + '/data/results/confusion_matrix_data/predictions_rf_train.csv')

predictions_df_test = pd.DataFrame(data=zip(truth_y_test, y_pred_test),
             columns=['truth_y_test', 'y_pred_test'])

predictions_df_test.to_csv(my_path + '/data/results/confusion_matrix_data/predictions_rf_test.csv')

# Save the model to a file
dump(model_rf, my_path + '/models/model_rf.joblib')

#endregion

#region HYPERPARAMETER TUNING - RANDOM SEARCH - XGBOOST

# Create the parameter search space
param_space = {
    # Randomly sample L2 penalty
    'lambda': randint(1, 10),

    # Randomly sample numbers
    'max_depth': randint(10, 100),

    # Normally distributed subsample, with mean .50 stddev 0.15, bounded between 0 and 1
    'subsample': truncnorm(a=0, b=1, loc=0.50, scale=0.15),

    # Uniform distribution for learning rate
    'eta': uniform(0.001, 0.3)
}

# Instantiate the model
ml_model = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss')

# Create the random search algorithm
random_search_xgb = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='accuracy'
)

# Train the random search algorithm
model_xgb = random_search_xgb.fit(truth_x_train, truth_y_train)

# Save training results to file
with open(my_path + '/doc/random_search_output.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('TRAINING INFORMATION - RANDOM SEARCH - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    pprint(model_xgb.cv_results_, stream=f)
    print('\n', file=f)

# Predict the training data
y_pred_train = model_xgb.predict(truth_x_train)

# Make predictions on the test data
y_pred_test = model_xgb.predict(truth_x_test)

# Print the metrics of the test results
with open(my_path + '/doc/random_search_output.txt', 'a') as f:
    print('TEST METRICS W/ BEST MODEL PARAMETERS - XGBOOST', file=f)
    print('Accuracy: %.3f' % metrics.accuracy_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('Precision: %.3f' % metrics.precision_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('Recall: %.3f' % metrics.recall_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('F1: %.3f' % metrics.f1_score(y_true=truth_y_test, y_pred=y_pred_test), file=f)
    print('ROC AUC: %.3f' % metrics.roc_auc_score(y_true=truth_y_test, y_score=y_pred_test), file=f)
    print('\n', file=f)

# Save all data for confusion matrix
predictions_df_train = pd.DataFrame(data=zip(truth_y_train, y_pred_train),
             columns=['truth_y_train', 'y_pred_train'])

predictions_df_train.to_csv(my_path + '/data/results/confusion_matrix_data/predictions_xgb_train.csv')

predictions_df_test = pd.DataFrame(data=zip(truth_y_test, y_pred_test),
             columns=['truth_y_test', 'y_pred_test'])

predictions_df_test.to_csv(my_path + '/data/results/confusion_matrix_data/predictions_xgb_test.csv')

# Save the model to a file
dump(model_xgb, my_path + '/models/model_xgb.joblib')

#endregion
