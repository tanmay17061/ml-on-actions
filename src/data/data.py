import pandas as pd

def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    y = df["quality"]
    df = df.drop(["quality"], axis=1)
    return df.to_numpy, y.to_numpy, list(df.columns)