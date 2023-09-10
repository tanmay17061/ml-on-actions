import logging
import pandas as pd
from ..data import load_df_from_csv,split_df_in_ratio,merge_dfs

def add_new_samples_pipeline(args):
    """

        assumes that the files `args.src_train_data_path` and `args.src_test_data_path` already exist, onto which `args.train_data_path` and `args.test_data_path` would be appended.

    """

    if bool(args.test_data_path) == bool(args.train_ratio):
        raise ValueError(f"exactly one of {args.test_data_path=} or {args.train_ratio} should be provided")
    
    new_train_df = load_df_from_csv(args.train_data_path)
    if args.test_data_path:
        new_test_df = load_df_from_csv(args.test_data_path)
    else:
        new_train_df,new_test_df = split_df_in_ratio(new_train_df,args.train_ratio)
    
    train_df = load_df_from_csv(args.src_train_data_path)
    test_df = load_df_from_csv(args.src_test_data_path)

    train_df = merge_dfs(train_df,new_train_df)
    test_df = merge_dfs(test_df,new_test_df)

    train_df.to_csv(args.src_train_data_path, index=False)
    test_df.to_csv(args.src_test_data_path, index=False)