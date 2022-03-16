# -*- coding: utf-8 -*-
"""basic_model.py docstring

Module created to do some basic modelling on our data.

Additionally it also saves the result as a seperate csv file and
the model as a pickle file for future use.
"""
import numpy as np
import pandas as pd

import joblib
import os
import os.path as op
import warnings


import tensorflow as tf
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import (
    Input,
    Flatten,
    Dropout,
    Activation,
    BatchNormalization
)
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(action="ignore")
np.random.seed(42)


def get_data(path):
    """Function to load the train and test data and to divide it into
    features and labels.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        path: StrOrBytesPath
            The path where from where the data is taken from.
    Returns
    ----------
    Training and testing data divided as 'X_train', 'X_test', 'y_train'
    and 'y_test'.
    """

    train_df = pd.read_csv(path + "/train.csv")
    test_df = pd.read_csv(path + "/test.csv")

    X_train = train_df.drop("labels", axis=1)
    y_train = train_df["labels"].copy()

    X_test = test_df.drop("labels", axis=1)
    y_test = test_df["labels"].copy()

    return X_train, X_test, y_train, y_test


def normalization(X_train, X_test):
    """Function to do some normalization on 'X_train' and 'X_test'.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        X_train: pd.Dataframe
            The features of training set.
        X_test: pd.Dataframe
            The features of testing set.
    Returns
    ----------
    Normalized 'X_train' and 'X_test'.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test


def baseline_test(X_train, X_test, y_train, y_test, models):
    """Function to do some baseline tests on our data.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        X_train: pd.Dataframe
            The features of training set.
        X_test: pd.Dataframe
            The features of testing set.
        y_train: pd.Dataframe
            Labels of the training set.
        y_test: pd.Dataframe
            Labels of the testing set.
        models: Dict
            Dictionary of the form {model_name: model} supplied by the
            use.
    Returns
    ----------
    The result of the basic modelling.
    """
    results = {}
    for model in models:
        models[model].fit(X_train, y_train)

        results[model] = classification_report(
            y_test,
            models[model].predict(X_test)
        )

    return results


def seq_model(X_train, X_test, y_train, y_test, path):
    """Function to do sequential modelling on our data.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        X_train: pd.Dataframe
            The features of training set.
        X_test: pd.Dataframe
            The features of testing set.
        y_train: pd.Dataframe
            Labels of the training set.
        y_test: pd.Dataframe
            Labels of the testing set.
        path: StrOrBytesPath
            Location where file is to be saved.
    Returns
    ----------
    The result of the basic modelling.
    """
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # one hot encode the target
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    with open("data_description.txt", "w") as f:
        f.write(f"{lb.classes_}")

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # New model
    model = Sequential()
    model.add(
        Conv1D(256, 8, padding="same", input_shape=(X_train.shape[1], 1))
    )  # X_train.shape[1] = No. of Columns
    model.add(Activation("relu"))
    model.add(Conv1D(256, 8, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding="same"))
    model.add(Activation("relu"))
    model.add(Conv1D(128, 8, padding="same"))
    model.add(Activation("relu"))
    model.add(Conv1D(128, 8, padding="same"))
    model.add(Activation("relu"))
    model.add(Conv1D(128, 8, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding="same"))
    model.add(Activation("relu"))
    model.add(Conv1D(64, 8, padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(14))
    model.add(Activation("softmax"))

    opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=80,
        verbose=1,
        validation_data=(X_test, y_test),
    )


    model_name = "seq_model.h5"
    model_path = op.join(path, model_name)
    model.save(model_path)

    # Save the model to disk
    model_json = model.to_json()
    with open("model_json.json", "w") as json_file:
        json_file.write(model_json)


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = get_data()
#     X_train, X_test = normalization(X_train, X_test)
#     baseline_test(X_train, X_test, y_train, y_test)
#     model = seq_model(X_train, X_test, y_train, y_test)
