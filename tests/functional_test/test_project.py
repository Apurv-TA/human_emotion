import os
import os.path as op
import sys

import numpy as np
import pandas as pd
import pytest


HERE = op.dirname(op.abspath("__file__"))
test_path = op.join(HERE, "..", "..", "src", "audio_emotion")
sys.path.append(test_path)

@pytest.fixture
def test_df():
    test = pd.read_csv("../../data/processed/test.csv").sample(20)

    return test


def test_model(test_df):
    assert isinstance(test_df, pd.DataFrame)

    X_test = test_df.drop("labels", axis=1)
    y_test = test_df["labels"].copy()

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

