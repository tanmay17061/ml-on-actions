import joblib
import logging
import os

from ..model import Model
from ..data import load_npy_from_csv
from sklearn.metrics import classification_report

def train_pipeline(args):
    if os.path.isfile(args.train_data_path) == os.path.isfile(args.model_pickle_path):
        raise ValueError(f"either one of `train_data_path` or `model_pickle_path` should be a valid file.")

    if os.path.isfile(args.train_data_path):
        train_data = load_npy_from_csv(args.train_data_path)
        model = Model()
        model.fit(X=train_data[0],y=train_data[1])

        y_pred = model.predict(X=train_data[0])
        y_true = train_data[1]
        with open("train_report.txt","wt") as f:
            print(classification_report(y_true,y_pred),file=f)
        joblib.dump(model,args.model_pickle_path)
    else:
        model = joblib.load(args.model_pickle_path)

    test_data = load_npy_from_csv(args.test_data_path)

    y_pred = model.predict(X=test_data[0])
    y_true = test_data[1]

    with open("test_report.txt","wt") as f:
        print(classification_report(y_true,y_pred),file=f)