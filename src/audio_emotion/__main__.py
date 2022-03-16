import os
import os.path as op

import numpy
import pandas

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from get_argument import argument
from logging_setup import configure_logger
from ingest_data import load_data
from feature_extraction import extract_split
from basic_model import (get_data, normalization, baseline_test, seq_model)
from score import class_report, load, gender_test, emotion_test


if __name__ == "__main__":
    args = argument()

    HERE = op.dirname(op.abspath(__file__))
    DATA = argument().data
    SAVE = argument().save
    DATA_PATH = op.join(HERE, DATA)
    PROCESSED_FOLDER = op.join(HERE, DATA, "processed")
    ARTIFACT_FOLDER = op.join(HERE, SAVE)

    MODELS = {
        "Logistic_regr": LogisticRegression(),
        "Decision_tree": DecisionTreeClassifier(),
        "Random_forest": RandomForestClassifier(
            max_features="log2",
            max_depth=10,
            max_leaf_nodes=100,
            min_samples_leaf=3,
            min_samples_split=20,
            n_estimators=22000,
            random_state=42
        )
    }

    if args.log_path:
        LOG_FILE = os.path.join(args.log_path, "custom_configure.log")
    else:
        LOG_FILE = None

    logger = configure_logger(
        log_file=LOG_FILE,
        console=args.no_console_log,
        log_level=args.log_level
    )

    logger.info("Starting the file run.")

    # ----------

    logger.info("Running ingest_data.py")
    ref = load_data(
        data_path=DATA_PATH
    )

    logger.debug(f"Total:\n{ref['labels'].value_counts()}")
    logger.debug("\n" + "-" * 100 + "\n")

    # ----------

    logger.info("Running feature_extraction.py")

    df, train, test = extract_split(ref=ref, data_folder=PROCESSED_FOLDER)

    logger.debug(f"df.shape -> {df.shape}")
    logger.info(f"Processed file saved in {PROCESSED_FOLDER}")
    logger.debug(f"train.shape -> {train.shape}")
    logger.debug(f"test.shape -> {test.shape}")

    # ----------

    logger.info("Running basic_model.py")

    X_train, X_test, y_train, y_test = get_data(path=PROCESSED_FOLDER)
    X_train, X_test = normalization(X_train, X_test)
    results = baseline_test(X_train, X_test, y_train, y_test, models=MODELS)

    for result in results:
        logger.debug(f"{result}\n{results[result]}")

    seq_model(X_train, X_test, y_train, y_test, path=ARTIFACT_FOLDER)
    logger.info(f"Saving the artifacts at {ARTIFACT_FOLDER}")

    # ----------

    logger.info("Starting score.py")

    pred_df = load(
        processed_path=PROCESSED_FOLDER,
        artifacts_path=ARTIFACT_FOLDER
    )
    logger.debug(
        f"Gender emotion classification\n{class_report(pred_df)}"
    )

    gender_df = gender_test(processed_path=PROCESSED_FOLDER)
    logger.debug(
        f"Gender classificastion:\n{class_report(gender_df)}"
    )

    emotion_df = emotion_test(processed_path=PROCESSED_FOLDER)
    logger.debug(
        f"Emotion classification:\n{class_report(emotion_df)}"
    )
