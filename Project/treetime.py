# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:42:24 2023

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
file_path = 'C:/Users/13410/OneDrive/Desktop/New folder/heston_training_set6.pkl'

# Read the DataFrame directly from the Pickle file
dataframe = pd.read_pickle(file_path)
results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Dataset_Size', 'MSE', 'MAE', 'Time'])

# List of dataset sizes to analyze
dataset_sizes = [i/100 for i in range(10, 105, 10)]  
best_params_rf_optuna={'n_estimators': 285, 'max_depth': 25, 'max_features': None}
best_params_gb_optuna={'n_estimators': 296, 'max_depth': 10, 'max_features': None}
best_params_dt_optuna={'max_depth': 31, 'min_samples_split': 4, 'min_samples_leaf': 3}
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
        start_time = time.time()
        pred_subset = model_subset.predict(X_test_subset)
        end_time = time.time()

        mse_subset = mean_squared_error(y_test_subset, pred_subset)
        mae_subset = mean_absolute_error(y_test_subset, pred_subset)

        # Add result for the model
        results_df = pd.concat([results_df,
                                pd.DataFrame({'Model': [model_name],
                                              'Parameters': [f'Optuna Tuned ({best_params_optuna})'],
                                              'Dataset_Size': [size],
                                              'MSE': [mse_subset],
                                              'MAE': [mae_subset],
                                              'Time': [end_time - start_time]})], ignore_index=True)

    # Save results to Excel with a unique filename for each model
    results_df.to_excel(f'C:/Users/13410/OneDrive/Desktop/New folder/dataset_size_analysis_{model_name.lower().replace(" ", "_")}_fixed_params.xlsx', index=False)

    # Plot MSE for each model
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Dataset_Size', y='MSE', data=results_df[results_df['Model'] == model_name])
    plt.title(f'{model_name} - Mean Squared Error Comparison')
    plt.savefig(f'C:/Users/13410/OneDrive/Desktop/New folder/mse_comparison_{model_name.lower().replace(" ", "_")}_fixed_params.png')
    plt.show()

    # Plot MAE for each model
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Dataset_Size', y='MAE', data=results_df[results_df['Model'] == model_name])
    plt.title(f'{model_name} - Mean Absolute Error Comparison')
    plt.savefig(f'C:/Users/13410/OneDrive/Desktop/New folder/mae_comparison_{model_name.lower().replace(" ", "_")}_fixed_params.png')
    plt.show()

    # Plot Time for each model
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Dataset_Size', y='Time', data=results_df[results_df['Model'] == model_name])
    plt.title(f'{model_name} - Prediction Time Comparison')
    plt.savefig(f'C:/Users/13410/OneDrive/Desktop/New folder/time_comparison_{model_name.lower().replace(" ", "_")}_fixed_params.png')
    plt.show()
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Loop through each model
for i, (model_name, _, _) in enumerate([('Random Forest', RandomForestRegressor, best_params_rf_optuna),
                                         ('Gradient Boosting', GradientBoostingRegressor, best_params_gb_optuna),
                                         ('Decision Tree', DecisionTreeRegressor, best_params_dt_optuna)]):
    model_results = results_df[results_df['Model'] == model_name]

    # Plot MSE
    sns.lineplot(x='Dataset_Size', y='MSE', data=model_results, ax=axes[i, 0])
    axes[i, 0].set_title(f'{model_name} - MSE Comparison')

    # Plot MAE
    sns.lineplot(x='Dataset_Size', y='MAE', data=model_results, ax=axes[i, 1])
    axes[i, 1].set_title(f'{model_name} - MAE Comparison')

    # Plot Time
    sns.lineplot(x='Dataset_Size', y='Time', data=model_results, ax=axes[i, 2])
    axes[i, 2].set_title(f'{model_name} - Prediction Time Comparison')

# Adjust layout
plt.tight_layout()

# Save the composite figure
plt.savefig('C:/Users/13410/OneDrive/Desktop/New folder/composite_comparison.png')

# Show the composite figure
plt.show()