# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:02:56 2023

@author: 13410
"""

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
file_path = 'C:/Users/13410/My Drive/New folder/treesmall/heston_training_set6.pkl'

# Read the DataFrame directly from the Pickle file
dataframe = pd.read_pickle(file_path)
results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Dataset_Size', 'Train_MSE', 'Train_MAE', 'Test_MSE', 'Test_MAE', 'Time'])

# List of dataset sizes to analyze
dataset_sizes = [10000/dataframe.shape[0], 20000/dataframe.shape[0], 40000/dataframe.shape[0], 80000/dataframe.shape[0], 100000/dataframe.shape[0]] 
best_params_rf_optuna = {'n_estimators': 285, 'max_depth': 25, 'max_features': None}
best_params_gb_optuna = {'n_estimators': 296, 'max_depth': 10, 'max_features': None}
best_params_dt_optuna = {'max_depth': 31, 'min_samples_split': 4, 'min_samples_leaf': 3}

# Add lists to store training losses
train_losses_mse = []
train_losses_mae = []

for model_name, model_class, best_params_optuna in [('Random Forest', RandomForestRegressor, best_params_rf_optuna),
                                                    ('Gradient Boosting', GradientBoostingRegressor, best_params_gb_optuna),
                                                    ('Decision Tree', DecisionTreeRegressor, best_params_dt_optuna)]:

    for size in tqdm(dataset_sizes):
        # Randomly pick a subset for testing
        subset_size = int(size * dataframe.shape[0])
        subset_indices = np.random.choice(dataframe.shape[0], size=subset_size, replace=False)
        subset_df = dataframe.iloc[subset_indices, :]

        # Train-test split for the subset
        x_subset = subset_df.iloc[:, :-1]
        y_subset = subset_df.iloc[:, -1]
        X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(x_subset, y_subset, test_size=0.2)

        # Model on Subset with Optuna Tuned Parameters
        model_subset = model_class(**best_params_optuna)
        start_time = time.time()
        model_subset.fit(X_train_subset, y_train_subset)
        end_time = time.time()

        # Calculate training loss after fitting
        train_pred = model_subset.predict(X_train_subset)
        train_loss_mse = mean_squared_error(y_train_subset, train_pred)
        train_loss_mae = mean_absolute_error(y_train_subset, train_pred)

        # Record training losses
        train_losses_mse.append(train_loss_mse)
        train_losses_mae.append(train_loss_mae)

        # Test the model
        pred_subset = model_subset.predict(X_test_subset)
        mse_subset = mean_squared_error(y_test_subset, pred_subset)
        mae_subset = mean_absolute_error(y_test_subset, pred_subset)

        # Add result for the model
        results_df = pd.concat([results_df,
                                pd.DataFrame({'Model': [model_name],
                                              'Parameters': [f'Optuna Tuned ({best_params_optuna})'],
                                              'Dataset_Size': [size],
                                              'Train_MSE': [train_loss_mse],
                                              'Train_MAE': [train_loss_mae],
                                              'Test_MSE': [mse_subset],
                                              'Test_MAE': [mae_subset],
                                              'Time': [end_time - start_time]})], ignore_index=True)

# Save results to Excel with a unique filename for each model
results_df.to_excel('C:/Users/13410/My Drive/New folder/treesmall/dataset_size_analysis_fixed_params.xlsx', index=False)

# Plot Training MSE for each model
plt.figure(figsize=(10, 6))
for model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    sns.lineplot(x='Dataset_Size', y='Train_MSE', data=results_df[results_df['Model'] == model_name], label=model_name)
plt.title('Training Mean Squared Error Comparison')
plt.savefig('C:/Users/13410/My Drive/New folder/treesmall/train_mse_comparison.png')
plt.show()

# Plot Training MAE for each model
plt.figure(figsize=(10, 6))
for model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    sns.lineplot(x='Dataset_Size', y='Train_MAE', data=results_df[results_df['Model'] == model_name], label=model_name)
plt.title('Training Mean Absolute Error Comparison')
plt.savefig('C:/Users/13410/My Drive/New folder/treesmall/train_mae_comparison.png')
plt.show()

# Plot Test MSE for each model
plt.figure(figsize=(10, 6))
for model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    sns.lineplot(x='Dataset_Size', y='Test_MSE', data=results_df[results_df['Model'] == model_name], label=model_name)
plt.title('Test Mean Squared Error Comparison')
plt.savefig('C:/Users/13410/My Drive/New folder/treesmall/test_mse_comparison.png')
plt.show()

# Plot Test MAE for each model
plt.figure(figsize=(10, 6))
for model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    sns.lineplot(x='Dataset_Size', y='Test_MAE', data=results_df[results_df['Model'] == model_name], label=model_name)
plt.title('Test Mean Absolute Error Comparison')
plt.savefig('C:/Users/13410/My Drive/New folder/treesmall/test_mae_comparison.png')
plt.show()
