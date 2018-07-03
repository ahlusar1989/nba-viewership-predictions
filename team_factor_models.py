"""This file uses the one-hot encoded team vectors to predict the total viewership
The theory behind this model is that each team has an international fanbase, which contributes to the game's total viewership
Therefore the coefficient for each team is their international fanbase relative to an average NBA Game
The intercept represents the number of international fans expected between two average teams

CONCLUSION: This model achieves an average cross-validation R^2 of 0.582 across 10 folds
    This model achieves an average cross-validation MAPE of 0.386 (NBA criteria for grading)
This appears to be a solid baseline model for TotalViewership"""
import pandas as pd
import numpy as np
import statsmodels.api as smi
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Read in file created from 'single_game_reformatting.py' script
input_filename = "training_set_games.csv"
df = pd.read_csv(input_filename)

# Get the columns to be used as inputs (Team indicator columns)
team_cols = [e for e in df.columns if "T-" in e]
input_cols = team_cols + ["Weeknight"]

# Get data in supervised learning format f:X -> y
X = df.loc[:, input_cols]
y = df.loc[:, "TotalViewers"]

# Initialize dictionary for tracking accuracy measures, where Mean- is using the training set mean for predictions
accuracy_dict = {"MSE":[],
                 "Mean-MSE":[],
                 "MAPE":[],
                 "Mean-MAPE":[]}

# Perform 10-Fold Cross-Validation
kfold = KFold(n_splits=10, shuffle=True)
for train_indices, test_indices in kfold.split(X):
    # Get the training data and fit a LinearRegression to the training data
    X_train, y_train = X.loc[train_indices, :], y.loc[train_indices]
    fold_model = LinearRegression()
    fold_model.fit(X=X_train, y=y_train)

    # Make predictions on the test data from the trained model
    X_test, y_test = X.loc[test_indices, :], y.loc[test_indices]
    predictions = fold_model.predict(X=X_test)

    # Calculate the Mean Square Error of the current test fold predictions
    error = predictions - y_test
    mean_error = y_train.mean() - y_test

    fold_mse = np.mean(error**2)
    accuracy_dict["MSE"].append(fold_mse)

    # Calculate the Sum of Squared Error of the current test fold predictions
    fold_sse = np.mean(mean_error**2)
    accuracy_dict["Mean-MSE"].append(fold_sse)

    # Calculate the Mean Absolute Percentage Error (MAPE) used by NBA to grade submissions
    fold_mape = np.mean(np.abs(error)/y_test)
    accuracy_dict["MAPE"].append(fold_mape)

    fold_mean_mape = np.mean(np.abs(mean_error)/y_test)
    accuracy_dict["Mean-MAPE"].append(fold_mean_mape)
# Converting the accuracy dictionary to a DataFrame
#   with columns as the accuracy measures and index as the fold number
results_df = pd.DataFrame.from_dict(accuracy_dict, orient='columns')
results_df["R2"] = results_df.apply(lambda row: 1.0 - row["MSE"]/row["Mean-MSE"], axis=1)
results_df["MAPE-R2"] = results_df.apply(lambda row: 1.0 - row["MAPE"]/row["Mean-MAPE"], axis=1)
print(results_df)
print(results_df.mean(axis=0).round(3))

# Using Statsmodels for output on full set to see if the coefficients make sense
# (they appear to, as CLE and GSW are highest, MEM and PHX are lowest)
#    Having weeknight games also appears to drop the amount of viewers
# The explanatory R-Squared is 0.603 with Adj R-Squared of 0.597,
#   the closeness of these to the CV-R2 indicates there is not much overfitting occurring here
Xc = smi.add_constant(X)
ols = smi.OLS(endog=y, exog=Xc)
results = ols.fit()
print(results.summary())
