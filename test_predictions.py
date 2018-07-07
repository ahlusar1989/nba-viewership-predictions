"""File for formatting the test.csv file and inserting predictions from final model
The model is saved as output from the team_factor_models.py script"""

import single_game_reformatting as sgr
import pandas as pd
import statsmodels.api as smi

# Original test set given by NBA
test_filename = "test_set.csv"
test_df = pd.read_csv(test_filename)

# Add the same features used in the training data model
formatted_test_df = sgr.create_team_vector(arg_df=test_df, split_seasons=True)
formatted_test_df = sgr.create_month_vector(single_game_df=formatted_test_df)
formatted_test_df["DayOfWeek"] = pd.to_datetime(formatted_test_df["Game_Date"]).dt.weekday_name
formatted_test_df["Weeknight"] = formatted_test_df["DayOfWeek"].apply(lambda cell: 1 if cell in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"] else 0)

# Add a column of 1s for the const in statsmodels (the intercept)
formatted_test_df['const'] = pd.Series([1] * formatted_test_df.shape[0])

# Load the model and get the features used by the model
saved_model = smi.regression.linear_model.RegressionResults.load("lm_7-6-2018.pkl")
features_used = saved_model.params.index.tolist()

# Take only the columns used by the trained model (and ensure all of those are selected)
reduced_test_df = formatted_test_df.loc[:, features_used]
assert reduced_test_df.shape[1] == len(features_used)

# Generate predictions on the test set using the trained model
predictions = saved_model.predict(reduced_test_df)

# Set the data into the desired column (specified in competition PDF)
test_df.loc[:, "Total_Viewers"] = predictions

# Output to the appropriate format (specified in the competition PDF)
# Temporarily using NCSU-ASW as the alphabetical initials of last names in group
test_df.to_csv("test_set_NCSU-ASW.csv", index=False)
