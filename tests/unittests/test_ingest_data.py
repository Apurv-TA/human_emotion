import os.path as op
import sys


import pandas as pd
import pytest


HERE = op.dirname(op.abspath("__file__"))
test_path = op.join(HERE, "..", "..", "src", "audio_emotion")
sys.path.append(test_path)
import ingest_data


@pytest.fixture
def test_df():
    df = pd.read_csv("../../data/processed/data.csv")
    return df


@pytest.fixture
def test_ref():
    ref = pd.read_csv("../../data/processed/ref.csv")


def test_load_data(test_ref):
    assert isinstance(test_ref, pd.DataFrame)


def test_final_data(test_df):
    assert isinstance(test_df, pd.DataFrame)
