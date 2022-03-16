# -*- coding: utf-8 -*-
"""score.py docstring

Module created for the purpose of determining how well our model performed on
test data.

Primarily our model was designed to determine 'gender_emotion' of the speaker,
here we have also added functions to determine its performance if only
gender or emotion classification is done.
"""


import json
import os
import os.path as op
import warnings

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential, model_from_json
from keras.utils import np_utils
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings(action="ignore")
np.random.seed(42)

def class_report(df):
    """Function to do classification report on our dataframe.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        df: pd.Dataframe
            The dataframe for which performance is to be determined.
    Returns
    ----------
    Classification report of the dataframe based on the labels present.
    """

    classes = df["actualvalues"].unique()
    classes.sort()
    return classification_report(
        df.actualvalues,
        df.predictedvalues,
        target_names=classes
    )


def load(processed_path, artifacts_path):
    """Function to load the data and model.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        processed_path: StrOrBytesPAth
            The path where the processed data needs to be saved.
        data_folder: StrOrBytesPath
            The path where the artifacts are needs to be saved.
    Returns
    ----------
    Prediction Dataframe containing the 'actualvalues' and 'predictionvalues'.
    """

    test = pd.read_csv(processed_path + "/test.csv")
    X_test = test.drop("labels", axis=1)
    y_test = test["labels"].copy()

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # one hot encode the target
    lb = LabelEncoder()
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    X_test = np.expand_dims(X_test, axis=2)

    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(artifacts_path + "/seq_model.h5")
 
    # Keras optimiser
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    score = loaded_model.evaluate(X_test, y_test, verbose=0)

    preds = loaded_model.predict(
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

    # Write out the predictions to disk
    finaldf.to_csv(processed_path + '/Predictions.csv', index=False)

    return finaldf


def gender_test(processed_path):
    """Function to predict gender classification on our data.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        processed_path: StrOrBytesPAth
            The path where the processed data is saved.
    Returns
    ----------
    The modified dataframe.
    """

    finaldf = pd.read_csv(processed_path + "/Predictions.csv")
    modidf = finaldf
    values = {
        'female_angry':'female',
        'female_disgust':'female',
        'female_fear':'female',
        'female_happy':'female',
        'female_sad':'female',
        'female_surprise':'female',
        'female_neutral':'female',
        'male_angry':'male',
        'male_fear':'male',
        'male_happy':'male',
        'male_sad':'male',
        'male_surprise':'male',
        'male_neutral':'male',
        'male_disgust':'male'
    }
    modidf['actualvalues'] = finaldf.actualvalues.replace(values)
    modidf['predictedvalues'] = finaldf.predictedvalues.replace(values)

    return modidf


def emotion_test(processed_path):
    """Function to do emotion classification on our data.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        processed_path: StrOrBytesPAth
            The path where the processed data is saved.
    Returns
    ----------
    The modified dataframe.
    """

    finaldf = pd.read_csv(processed_path + "/Predictions.csv")
    modidf = finaldf
    values = {
        'female_angry':'angry',
        'female_disgust':'disgust',
        'female_fear':'fear',
        'female_happy':'happy',
        'female_sad':'sad',
        'female_surprise':'surprise',
        'female_neutral':'neutral',
        'male_angry':'angry',
        'male_fear':'fear',
        'male_happy':'happy',
        'male_sad':'sad',
        'male_surprise':'surprise',
        'male_neutral':'neutral',
        'male_disgust':'disgust'
    }
    modidf['actualvalues'] = modidf.actualvalues.replace(values)
    modidf['predictedvalues'] = modidf.predictedvalues.replace(values)

    return modidf
