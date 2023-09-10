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