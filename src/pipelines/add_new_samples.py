import logging
import pandas as pd

def add_new_samples_pipeline(args):
    """
        assumes that the directory `REPO_ROOT/data/dataset` already contains the
        original data CSVs (`train.csv` and `test.csv`) that would be appended onto.
    """

    if args.test_data_path and args.train_ratio:
        raise ValueError(f"only one of {args.test_data_path=} or {args.train_ratio} should be provided")

    logging.info("inside add_new_samples_pipeline")