import argparse
import logging
from mlonactions.pipelines import train_pipeline, add_new_samples_pipeline, data_and_model_monitoring_pipeline

logging.basicConfig(level=logging.DEBUG)

def build_train_subparser(parser):
    parser.add_argument("--train_data_path", type=str, help="path of train dataset.", default="")
    parser.add_argument("--test_data_path", type=str, help="path of test dataset.", required=True)
    parser.add_argument("--model_joblib_path", type=str, help="path to: 1. save model at if `train_data_path` is provided, or, 2. load model from for testing.", required=True)

def build_add_new_samples_parser(parser):
    parser.add_argument("--train_data_path", type=str, help="path of new train dataset.")
    parser.add_argument("--test_data_path", type=str, help="path of new test dataset.")
    parser.add_argument("--src_train_data_path", type=str, help="path of original train dataset.")
    parser.add_argument("--src_test_data_path", type=str, help="path of original test dataset.")
    parser.add_argument("--train_ratio", type=float, help="ratio of train-split in train-test. eg- set this as `1` to remove test or `0` to remove train.", default=None)

def build_data_and_model_monitoring_parser(parser):
    parser.add_argument("--reference_data_path", type=str, help="path of reference dataset.")
    parser.add_argument("--current_data_path", type=str, help="path of current dataset.")
    parser.add_argument("--model_joblib_path", type=str, help="path to load model from, for testing.", required=True)

def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="types of commands supported by this cli tool", required=True)

    train_parser = subparsers.add_parser("train")
    build_train_subparser(train_parser)
    add_new_samples_parser = subparsers.add_parser("add_new_samples")
    build_add_new_samples_parser(add_new_samples_parser)
    data_and_model_monitoring_parser = subparsers.add_parser("data_and_model_monitoring")
    build_data_and_model_monitoring_parser(data_and_model_monitoring_parser)

    args = parser.parse_args()
    if args.command == "train":
        train_pipeline(args)
    if args.command == "add_new_samples":
        add_new_samples_pipeline(args)
    if args.command == "data_and_model_monitoring":
        data_and_model_monitoring_pipeline(args)

if __name__ == "__main__":
    main()