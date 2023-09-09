import argparse
import logging
from pipelines import train_pipeline, add_new_samples_pipeline

logging.basicConfig(level=logging.DEBUG)

def build_train_subparser(parser):
    parser.add_argument("--data_path", type=str, help="path of train+val dataset", required=True)
    parser.add_argument("--train_ratio", type=float, help="ratio of train-split in train-val. eg- set this as 1 to remove validation.", required=True)

def build_add_new_samples_parser(parser):
    parser.add_argument("--train_data_path", type=str, help="path of new train dataset.")
    parser.add_argument("--test_data_path", type=str, help="path of new test dataset.")
    parser.add_argument("--train_ratio", type=float, help="ratio of train-split in train-test. eg- set this as `1` to remove test or `0` to remove train.")

def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="types of commands supported by this cli tool", required=True)

    train_parser = subparsers.add_parser("train")
    build_train_subparser(train_parser)
    add_new_samples_parser = subparsers.add_parser("add_new_samples")
    build_add_new_samples_parser(add_new_samples_parser)

    args = parser.parse_args()
    if args.command == "train":
        train_pipeline(args)
    if args.command == "add_new_samples":
        add_new_samples_pipeline(args)

if __name__ == "__main__":
    main()