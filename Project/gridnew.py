# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:30:06 2023

@author: 13410
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm for the progress bar
from sklearn.metrics import make_scorer

train_file_path = 'C:/Users/13410/OneDrive/Desktop/New folder/training_set.csv'
val_file_path = 'C:/Users/13410/OneDrive/Desktop/New folder/val_set.csv'
test_file_path = 'C:/Users/13410/OneDrive/Desktop/New folder/test_set.csv'

# Read the DataFrames from the new CSV files
train_dataframe = pd.read_csv(train_file_path)
val_dataframe = pd.read_csv(val_file_path)
test_dataframe = pd.read_csv(test_file_path)

# Randomly select 80,000 samples from the training set as the training subset
subset_size_train = 80000
subset_indices_train = np.random.choice(train_dataframe.shape[0], size=subset_size_train, replace=False)
X_train_subset = train_dataframe.iloc[subset_indices_train, :-1]
y_train_subset = train_dataframe.iloc[subset_indices_train, -1]

# Randomly select 20,000 samples from the validation set as the testing subset
subset_size_val = 20000
subset_indices_val = np.random.choice(val_dataframe.shape[0], size=subset_size_val, replace=False)
X_test_subset = val_dataframe.iloc[subset_indices_val, :-1]
y_test_subset = val_dataframe.iloc[subset_indices_val, -1]

# Full training set from the entire training set
X_train_full = train_dataframe.iloc[:, :-1]
y_train_full = train_dataframe.iloc[:, -1]

# Full testing set from the entire test set
X_test_full = test_dataframe.iloc[:, :-1]
y_test_full = test_dataframe.iloc[:, -1]
results_df = pd.DataFrame(columns=['Model', 'Parameters', 'MSE', 'MAE', 'Time'])
# Decision Tree Grid Search on Subset
dt_regressor_subset = DecisionTreeRegressor()  # Create a Decision Tree Regressor

param_grid_dt_subset = {
    'max_depth': [5, 10, 15, 20, 25, 30, 35, 40],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 8],
}

scoring_function = make_scorer(mean_absolute_error, greater_is_better=False)

# Use GridSearchCV to find the best parameters
grid_search_dt_subset = GridSearchCV(dt_regressor_subset, param_grid_dt_subset, scoring=scoring_function, cv=5, n_jobs=-1)
grid_search_dt_subset.fit(X_train_subset, y_train_subset)

# Get the best parameters from the grid search
best_params_dt_subset = grid_search_dt_subset.best_params_

# Create a DataFrame to store grid search results
grid_search_results_dt = pd.DataFrame(columns=['Parameters', 'MSE', 'MAE'])

# Use tqdm to show a progress bar
for param_combination in tqdm(grid_search_dt_subset.cv_results_['params']):
    dt_regressor_subset.set_params(**param_combination)

    dt_regressor_subset.fit(X_train_subset, y_train_subset)
    start_time_dt_subset = time.time()
    dt_pred_subset = dt_regressor_subset.predict(X_test_subset)
    end_time_dt_subset = time.time()

    mse_subset = mean_squared_error(y_test_subset, dt_pred_subset)
    mae_subset = mean_absolute_error(y_test_subset, dt_pred_subset)

    # Add results to the DataFrame
    grid_search_results_dt = pd.concat([grid_search_results_dt,
                                       pd.DataFrame({'Parameters': [param_combination],
                                                     'MSE': [mse_subset],
                                                     'MAE': [mae_subset]})], ignore_index=True)

# Save grid search results to Excel
grid_search_results_dt.to_excel('grid_search_results1_dt.xlsx', index=False)

# Get best parameters from subset grid search
best_params_dt_subset = grid_search_results_dt.loc[grid_search_results_dt['MSE'].idxmin(), 'Parameters']

# Decision Tree on Full Dataset with Tuned Parameters
dt_regressor_full = DecisionTreeRegressor(**best_params_dt_subset)
dt_regressor_full.fit(X_train_full, y_train_full)
start_time_dt_full = time.time()

dt_pred_full = dt_regressor_full.predict(X_test_full)
end_time_dt_full = time.time()

dt_mse_full = mean_squared_error(y_test_full, dt_pred_full)
dt_mae_full = mean_absolute_error(y_test_full, dt_pred_full)
# Add Tuned result for Full Dataset
results_df = pd.concat([results_df,
                       pd.DataFrame({'Model': ['Decision Tree'],
                                     'Parameters': [f'Tuned ({best_params_dt_subset}) on Subset, Full Dataset'],
                                     'MSE': [dt_mse_full],
                                     'MAE': [dt_mae_full],
                                     'Time': [end_time_dt_full - start_time_dt_full]})], ignore_index=True)

# Random Forest Grid Search on Subset
rf_regressor_subset = RandomForestRegressor()

param_grid_rf_subset = {
    'max_depth': [5, 10, 15, 20, 25, 30, 35, 40],
    'n_estimators': [100, 150, 200, 250, 300],
    'max_features': [None, 'sqrt', 'log2'],
}
scoring_function = make_scorer(mean_absolute_error, greater_is_better=False)

# Use GridSearchCV to find the best parameters
grid_search_rf_subset = GridSearchCV(rf_regressor_subset, param_grid_rf_subset, scoring=scoring_function, cv=5, n_jobs=-1)
grid_search_rf_subset.fit(X_train_subset, y_train_subset)

# Get the best parameters from the grid search
best_params_rf_subset = grid_search_rf_subset.best_params_

# Create a DataFrame to store grid search results
grid_search_results_rf = pd.DataFrame(columns=['Parameters', 'MSE', 'MAE'])

# Use tqdm to show a progress bar
for param_combination in tqdm(grid_search_rf_subset.cv_results_['params']):
    rf_regressor_subset.set_params(**param_combination)

    
    rf_regressor_subset.fit(X_train_subset, y_train_subset)
    start_time_rf_subset = time.time()
    rf_pred_subset = rf_regressor_subset.predict(X_test_subset)
    end_time_rf_subset = time.time()

    mse_subset = mean_squared_error(y_test_subset, rf_pred_subset)
    mae_subset = mean_absolute_error(y_test_subset, rf_pred_subset)

    # Add results to the DataFrame
    grid_search_results_rf = pd.concat([grid_search_results_rf,
                                         pd.DataFrame({'Parameters': [param_combination],
                                                       'MSE': [mse_subset],
                                                       'MAE': [mae_subset]})], ignore_index=True)

# Save grid search results to Excel
grid_search_results_rf.to_excel('grid_search_results1_rf.xlsx', index=False)

# Get best parameters from subset grid search
best_params_rf_subset = grid_search_results_rf.loc[grid_search_results_rf['MSE'].idxmin(), 'Parameters']

# Random Forest on Full Dataset with Tuned Parameters
rf_regressor_full = RandomForestRegressor(**best_params_rf_subset)
rf_regressor_full.fit(X_train_full, y_train_full)
start_time_rf_full = time.time()

rf_pred_full = rf_regressor_full.predict(X_test_full)
end_time_rf_full = time.time()

rf_mse_full = mean_squared_error(y_test_full, rf_pred_full)
rf_mae_full = mean_absolute_error(y_test_full, rf_pred_full)
# Add Tuned result for Full Dataset
results_df = pd.concat([results_df,
                       pd.DataFrame({'Model': ['Random Forest'],
                                     'Parameters': [f'Tuned ({best_params_rf_subset}) on Subset, Full Dataset'],
                                     'MSE': [rf_mse_full],
                                     'MAE': [rf_mae_full],
                                     'Time': [end_time_rf_full - start_time_rf_full]})], ignore_index=True)

# Gradient Boosting Grid Search on Subset
gb_regressor_subset = GradientBoostingRegressor()

param_grid_gb_subset = {
    'max_depth': [3, 5, 7, 9, 11, 13, 15],
    'n_estimators': [100, 150, 200, 250, 300],
    'max_features': [None, 'sqrt', 'log2'],
}

# Create a DataFrame to store grid search results
grid_search_results_gb = pd.DataFrame(columns=['Parameters', 'MSE', 'MAE'])

# Use GridSearchCV to find the best parameters
grid_search_gb_subset = GridSearchCV(gb_regressor_subset, param_grid_gb_subset, scoring=scoring_function, cv=5, n_jobs=-1)
grid_search_gb_subset.fit(X_train_subset, y_train_subset)

# Get the best parameters from the grid search
best_params_gb_subset = grid_search_gb_subset.best_params_

# Use tqdm to show a progress bar
for param_combination in tqdm(grid_search_gb_subset.cv_results_['params']):
    gb_regressor_subset.set_params(**param_combination)

    
    gb_regressor_subset.fit(X_train_subset, y_train_subset)
    start_time_gb_subset = time.time()
    gb_pred_subset = gb_regressor_subset.predict(X_test_subset)
    end_time_gb_subset = time.time()

    mse_subset = mean_squared_error(y_test_subset, gb_pred_subset)
    mae_subset = mean_absolute_error(y_test_subset, gb_pred_subset)

    # Add results to the DataFrame
    grid_search_results_gb = pd.concat([grid_search_results_gb,
                                         pd.DataFrame({'Parameters': [param_combination],
                                                       'MSE': [mse_subset],
                                                       'MAE': [mae_subset]})], ignore_index=True)

# Save grid search results to Excel
grid_search_results_gb.to_excel('grid_search_results1_gb.xlsx', index=False)

# Get best parameters from subset grid search
best_params_gb_subset = grid_search_results_gb.loc[grid_search_results_gb['MSE'].idxmin(), 'Parameters']

# Gradient Boosting on Full Dataset with Tuned Parameters
gb_regressor_full = GradientBoostingRegressor(**best_params_gb_subset)

gb_regressor_full.fit(X_train_full, y_train_full)
start_time_gb_full = time.time()

gb_pred_full = gb_regressor_full.predict(X_test_full)
end_time_gb_full = time.time()

gb_mse_full = mean_squared_error(y_test_full, gb_pred_full)
gb_mae_full = mean_absolute_error(y_test_full, gb_pred_full)
# Add Tuned result for Full Dataset
results_df = pd.concat([results_df,
                       pd.DataFrame({'Model': ['Gradient Boosting'],
                                     'Parameters': [f'Tuned ({best_params_gb_subset}) on Subset, Full Dataset'],
                                     'MSE': [gb_mse_full],
                                     'MAE': [gb_mae_full],
                                     'Time': [end_time_gb_full - start_time_gb_full]})], ignore_index=True)

# Save results to Excel
results_df.to_excel('model_results_with_grid_search_mse_mae_prediction_time1.xlsx', index=False)

# Plot MSE
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', hue='Parameters', data=results_df)
plt.title('Mean Squared Error Comparison')
plt.savefig('mse_comparison_grid_search_mse_mae_prediction_time1.jpg')
plt.show()

# Plot MAE
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', hue='Parameters', data=results_df)
plt.title('Mean Absolute Error Comparison')
plt.savefig('mae_comparison_grid_search_mse_mae_prediction_time1.jpg')
plt.show()

# Plot Time
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Time', hue='Parameters', data=results_df)
plt.title('Prediction Time Comparison')
plt.savefig('time_comparison_grid_search_mse_mae_prediction_time11`.jpg')
plt.show()
