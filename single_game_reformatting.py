import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def create_single_game_df(og_df):
    """Create a DataFrame where each row is a single game, and each country's viewership is a column
    Will do this by adding new column for a Game_ID and taking transpose at the end"""
    all_countries = ["C-%s" % e for e in og_df["Country"].unique().tolist()]
    df_index = ["Season", "Home_Team", "Away_Team", "Game_Date"] + all_countries
    new_df = pd.DataFrame(index=df_index)
    for game_id in og_df["Game_ID"].unique():
        # Define the new series using the same index as the dataframe it will be concatenated to
        game_series = pd.Series(index=df_index)
        # Isolate the rows relevant for the current game
        game_rows = og_df.loc[og_df["Game_ID"] == game_id, :]

        # Since much of the data is repeated, get important information from the first row of the game
        game_series.loc[["Season", "Home_Team", "Away_Team", "Game_Date"]] = game_rows.iloc[0, :].loc[["Season", "Home_Team", "Away_Team", "Game_Date"]]

        # Add each country's viewership to the new series in the appropriate index location
        # NOTE: There is probably a more efficient way to do this
        for i,row in game_rows.iterrows():
            game_series.loc["C-%s" % row["Country"]] = row["Rounded Viewers"]

        # Add the game's data as a new column in the full df
        new_df[game_id] = game_series

    # Currently the columns are game_ids and the rows are the information/data,
    #   swap these to be more compatible with supervised learning methods using transpose
    new_df = new_df.T

    # Add the total viewers by summing over the country columns
    country_cols = [e for e in new_df.columns if "C-" in e]
    new_df["TotalViewers"] = new_df.loc[:, country_cols].sum(axis=1)
    return new_df


def create_team_vector(arg_df, split_seasons=False):
    """Function for creating a two-hot vector with bits representing the presence of specific teams in the game
        If split_seasons == True, treat team identifiers as unique per season"""
    single_game_df = arg_df.copy()
    # Converting teams to integer encoding
    team_encoder = LabelEncoder()
    if split_seasons:
        # Add a suffix that is the last two characters of the Season (aka ATL during 2016-2017 becomes ATL17)
        single_game_df["Home_Team"] = single_game_df.apply(lambda row: row["Home_Team"] + row["Season"][-2:], axis=1)
        single_game_df["Away_Team"] = single_game_df.apply(lambda row: row["Away_Team"] + row["Season"][-2:], axis=1)
    team_encoder.fit(np.ravel([single_game_df["Home_Team"], single_game_df["Away_Team"]]))
    single_game_df["Home_Team_Code"] = team_encoder.transform(single_game_df["Home_Team"])
    single_game_df["Away_Team_Code"] = team_encoder.transform(single_game_df["Away_Team"])

    vector_encoder = OneHotEncoder(sparse=False)
    away_vectors = vector_encoder.fit_transform(single_game_df["Away_Team_Code"].values.reshape(-1, 1))
    home_vectors = vector_encoder.fit_transform(single_game_df["Home_Team_Code"].values.reshape(-1, 1))
    comb_vector = away_vectors + home_vectors
    comb_df = pd.DataFrame(data=comb_vector, index=single_game_df.index, columns=["T-%s" % e for e in team_encoder.classes_])

    merged_df = pd.concat([single_game_df, comb_df], axis=1)
    return merged_df


def create_month_vector(single_game_df):
    """Function for adding an indicator for which month of the NBA season the game is played in
    Doing this because the NFL investigation found that Fall games had large crowds"""
    label_encoder = LabelEncoder()
    single_game_df["Month"] = single_game_df["Game_Date"].apply(lambda cell: cell.split("/")[0])
    single_game_df["MonthCode"] = label_encoder.fit_transform(single_game_df["Month"])

    vector_encoder = OneHotEncoder(sparse=False)
    month_vector = vector_encoder.fit_transform(single_game_df["MonthCode"].values.reshape(-1, 1))
    as_df = pd.DataFrame(data=month_vector, index=single_game_df.index, columns=["M-%s" % e for e in label_encoder.classes_])

    merged_df = pd.concat([single_game_df, as_df], axis=1)
    return merged_df

if __name__ == "__main__":
    # Original training set given by NBA
    training_filename = "training_set.csv"
    train_df = pd.read_csv(training_filename)

    full_df = create_single_game_df(og_df=train_df)
    full_df = full_df.fillna(0)
    # Intermediate saving since create_single_game_df takes a few seconds to run
    # full_df.to_csv("training_set_games_intermediate.csv", index=True)
    # full_df = pd.read_csv("training_set_games_intermediate.csv", index_col=0)
    team_df = create_team_vector(full_df, split_seasons=True)
    team_df = create_month_vector(team_df)


    # Adding day of week and indicator if that day is a weeknight (or not)
    team_df["DayOfWeek"] = pd.to_datetime(team_df["Game_Date"]).dt.weekday_name
    team_df["Weeknight"] = team_df["DayOfWeek"].apply(lambda cell: 1 if cell in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"] else 0)

    # Outputting the revised data frame in a 'per game' format
    team_df.to_csv("training_set_games_split.csv", index=True, index_label="Game_ID")



