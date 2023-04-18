"""
Feature selection using default XGBoost method, summarizing feature selection results, and visualizing.

RESOURCES
- Scikit-learn documentation
- Numpy documentation
- Pandas documentation
- https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
- https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
- https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
- https://stats.stackexchange.com/questions/264254/rerunning-with-only-important-features-doesnt-change-model-output#:~:text=As%20iws%20said%2C%20xgboost%20does,will%20not%20affect%20the%20results.
- https://stackoverflow.com/questions/12235552/r-function-rep-in-python-replicates-elements-of-a-list-vector
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

# Import predictions and true labels
predictions_xgb_test = pd.read_csv(my_path + '/data/results/confusion_matrix_data/predictions_xgb_test.csv')
predictions_xgb_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/predictions_xgb_train.csv')

# Import Reddit features and labels for error analysis
truth_x_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_x_train.csv')
truth_x_test = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_x_test.csv')
truth_y_train = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_y_train.csv', header=None)
truth_y_test = pd.read_csv(my_path + '/data/results/confusion_matrix_data/truth_y_test.csv', header=None)

#region DATA PREPARATION

# Prepare data for error tree model
truth_x_train1 = truth_x_train.drop(['Unnamed: 0', 'index'], axis=1)
truth_x_test1 = truth_x_test.drop(['Unnamed: 0', 'index'], axis=1)

# Drop first row on y data
truth_y_train = truth_y_train.iloc[1:]
truth_y_test = truth_y_test.iloc[1:]

# Get the feature names
feature_names = truth_x_train1.columns

# Transform data into matrices and vectors
truth_x_train1 = truth_x_train1.values
truth_x_test1 = truth_x_test1.values
truth_y_train1 = truth_y_train[1].values
truth_y_test1 = truth_y_test[1].values

# Convert to integer
truth_y_train1 = truth_y_train1.astype(int)
truth_y_test1 = truth_y_test1.astype(int)

# Instantiate the standard scaler
sc = StandardScaler()

# Standardize the feature matrix
truth_x_train1 = sc.fit_transform(truth_x_train1)
truth_x_test1 = sc.transform(truth_x_test1)

#endregion

#region FEATURE IMPORTANCE

# Trouble loading XGBoost model with joblib, so retrain with best hyperparameters
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.17413992398042408,
                    reg_lambda=3, max_depth=33, subsample=0.5578680145837858)

# Fit the XGBoost classifier to the data
xgb.fit(truth_x_train1, truth_y_train1)

# Get the feature importance
feature_importance = xgb.get_booster().get_score(importance_type='gain')

# Put feature importance into data frame
features_df = pd.DataFrame.from_dict(feature_importance, orient='index')

# Move the index to a column and clean feature name for subsetting
features_df = features_df.reset_index(level=0)
features_df = features_df.rename(columns={0: 'info_gained', 'index': 'feature'})
features_df['feature'] = features_df['feature'].str.replace('f', '')

# Sort the feature by importance gained
features_df = features_df.sort_values(by='info_gained', ascending=False)

#endregion

#region FIND THE FEATURES BASED ON IMPORTANCE

# Create a dataframe of feature names
feature_names_df = (
    feature_names.to_frame()
    .reset_index(level=0)
)

# Drop one of the columns and rename
feature_names_df = (
    feature_names_df.drop(0, axis=1)
    .rename(columns={'index': 'feature_names'})
    .reset_index(level=0)
    .rename(columns={'index': 'feature'})
)

# Create a feature category
feature_names_df['category'] = np.repeat(np.array(['none']), feature_names_df.shape[0])

# Assign feature topics

# Clinical keywords
feature_names_df['category'] = np.where(feature_names_df['feature_names'] == 'clinical_keywords', 'clinical_keywords',
                                        feature_names_df['category'])

# Sentiment
feature_names_df['category'] = np.where(feature_names_df['feature_names'] == 'sentiment_valence', 'sentiment_valence',
                                        feature_names_df['category'])

# Psycholinguistic attributes
feature_names_df['category'] = (
    np.where(feature_names_df['feature_names'].isin(feature_names_df['feature_names']
                                                    .values[2:95]),
             'psycholinguistic', feature_names_df['category'])
)

# n-grams
feature_names_df['category'] = (
    np.where(feature_names_df['feature_names'].isin(feature_names_df['feature_names']
                                                    .values[95:643]),
             'n_grams', feature_names_df['category'])
)

# DASS
feature_names_df['category'] = (
    np.where(feature_names_df['feature_names'].isin(feature_names_df['feature_names']
                                                    .values[643:647]),
             'dass', feature_names_df['category'])
)

# Word embeddings
feature_names_df['category'] = (
    np.where(feature_names_df['feature_names'].isin(feature_names_df['feature_names']
                                                    .values[647:947]),
             'embedding', feature_names_df['category'])
)

# Copy the dataframe for comparisons
feature_names_df_original = feature_names_df

# Ensure identity columns are same type
feature_names_df['feature'] = pd.to_numeric(feature_names_df['feature'])
features_df['feature'] = pd.to_numeric(features_df['feature'])

# Filter the data frame
feature_names_df = (
    feature_names_df.merge(features_df, on='feature')
    .sort_values(by='info_gained', ascending=False)
)

# Top ten features
print(feature_names_df.iloc[0:10, :])

# Save the feature importance data frame
feature_names_df.to_csv(my_path + '/data/results/feature_selection/all_features_by_importance.csv')

#endregion

#region SUMMARIZE FEATURE IMPORTANCE

# Count the number of features used before XGBoost feature selection
feature_count_original = feature_names_df_original['category'].value_counts()
feature_count_original = pd.DataFrame(feature_count_original)

# Count the number of features used in each feature category after XGBoost feature selection
feature_count = feature_names_df['category'].value_counts()
feature_count = pd.DataFrame(feature_count)

# Rename for merge
feature_count_original = feature_count_original.rename(columns={'category': 'original_n'})
feature_count = feature_count.rename(columns={'category': 'feat_select_n'})

# Concatenate the dataframes
count_df = pd.concat([feature_count_original, feature_count], axis=1)

# Percentage of features used in XGBoost
count_df['percent_used'] = count_df['feat_select_n'] / count_df['original_n']
count_df = count_df.reset_index(level=0)

# Get the average information gained
avg_feature_importance = feature_names_df.groupby('category')['info_gained'].mean()
avg_feature_importance = pd.DataFrame(avg_feature_importance)
avg_feature_importance = avg_feature_importance.reset_index(level=0)

# Export for visualization in R
count_df.to_csv(my_path + '/data/results/feature_selection/number_features_selected.csv')
avg_feature_importance.to_csv(my_path + '/data/results/feature_selection/avg_feature_importance.csv')

#endregion
