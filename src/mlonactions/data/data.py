from sklearn.model_selection import train_test_split

import pandas as pd
import os

def load_df_from_csv(csv_path):
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, index_col=False)
    else:
        df = None
    return df


def merge_dfs(*dfs):
    return pd.concat(dfs, ignore_index=True)


def load_npy_from_csv(csv_path):

    df = load_df_from_csv(csv_path)
    y = df["quality"]
    df = df.drop(["quality"], axis=1)
    return df.to_numpy, y.to_numpy, list(df.columns)


def split_df_in_ratio(df, split_ratio):
    if split_ratio == 1.0:
        # sklearn's train_test_split does not work when split_ratio = 1.0
        return df, None
    df1,df2 = train_test_split(df, train_size=split_ratio, shuffle=False)
    if len(df1) == 0: df1 = None
    if len(df2) == 0: df2 = None
    return df1,df2