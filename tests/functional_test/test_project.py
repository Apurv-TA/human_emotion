import os
import os.path as op
import sys

import numpy as np
import pandas as pd
import pytest
import keras
import tensorflow as tf
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

HERE = op.dirname(op.abspath("__file__"))
test_path = op.join(HERE, "..", "..", "src", "audio_emotion")
sys.path.append(test_path)

@pytest.fixture
def test_df():
    test = pd.read_csv("../../data/processed/test.csv").sample(20)

    return test


@pytest.fixture
def get_model():
    json_file = open(test_path + "/model_json.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("../../artifacts/seq_model.h5")

    # Keras optimiser
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    return loaded_model


def test_model(test_df, get_model):
    X_test = test_df.drop("labels", axis=1)
    y_test = test_df["labels"].copy()

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # one hot encode the target
    lb = LabelEncoder()
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    X_test = np.expand_dims(X_test, axis=2)

    score = get_model.evaluate(X_test, y_test, verbose=0)

    preds = get_model.predict(
        X_test,
        batch_size=16,
        verbose=1
    )

    preds=preds.argmax(axis=1)

    # predictions 
    preds = preds.astype(int).flatten()
    preds = (lb.inverse_transform((preds)))
    preds = pd.DataFrame({'predictedvalues': preds})

    # Actual labels
    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform((actual)))
    actual = pd.DataFrame({'actualvalues': actual})

    # Lets combined both of them into a single dataframe
    finaldf = actual.join(preds)

    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(finaldf, pd.DataFrame)
