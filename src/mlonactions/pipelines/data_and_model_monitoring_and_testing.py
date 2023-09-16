import logging
import os
import pandas as pd
from ..data import load_df_from_csv,load_npy_from_csv
from evidently import ColumnMapping
import joblib

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPreset

from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import NoTargetPerformanceTestPreset
from evidently.test_preset import DataQualityTestPreset
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataDriftTestPreset
from evidently.test_preset import RegressionTestPreset
from evidently.test_preset import MulticlassClassificationTestPreset
from evidently.test_preset import BinaryClassificationTopKTestPreset
from evidently.test_preset import BinaryClassificationTestPreset

def data_and_model_monitoring_and_testing_pipeline(args):
    """
        args.reference_data_path, args.current_data_path, args.model_joblib_path
    """

    assert os.path.isfile(args.current_data_path), f""
    
    reference_df = load_df_from_csv(args.reference_data_path)
    reference_data = load_npy_from_csv(args.reference_data_path)
    current_df = load_df_from_csv(args.current_data_path)
    current_data = load_npy_from_csv(args.current_data_path)
    model = joblib.load(args.model_joblib_path)

    reference_pred = model.predict(X=reference_data[0])
    current_pred = model.predict(X=current_data[0])

    reference_df["target"] = reference_df["quality"]
    current_df["target"] = current_df["quality"]

    del reference_df["quality"],current_df["quality"]

    reference_df["prediction"] = reference_pred
    current_df["prediction"] = current_pred

    del reference_pred,current_pred

    column_mapping = ColumnMapping(
        numerical_features=list(set(reference_df.columns) - set(["target","prediction"])),
        target="target",
        prediction="prediction",
        id=None
    )

    report = Report(metrics=[
    DataQualityPreset(),
    ClassificationPreset()
    ])

    report.run(current_data=current_df, reference_data = reference_df, column_mapping=column_mapping)
    report.save_html("monitoring_report.html")

    tests = TestSuite(tests=[
    DataStabilityTestPreset(),
    DataQualityTestPreset(),
    DataDriftTestPreset(),
    MulticlassClassificationTestPreset(),
    ])
    tests.run(current_data=current_df, reference_data = reference_df, column_mapping=column_mapping)
    tests.save_html("testing_report.html")
    tests_report_dict = tests.as_dict()
    
    failed_tests = tests_report_dict["summary"]["failed_tests"]
    total_tests = tests_report_dict["summary"]["total_tests"]
    assert tests_report_dict["summary"]["all_passed"], f"({failed_tests}/{total_tests}) DATA+MODEL TESTS FAILED. Please look at the test reports and resolve."