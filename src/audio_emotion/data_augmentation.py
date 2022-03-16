import glob
import json
import os
import os.path as op
import pickle

import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import (LSTM, Activation, AveragePooling1D,
                          BatchNormalization, Conv1D, Dense, Dropout,
                          Embedding, Flatten, Input, MaxPooling1D,)
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, to_categorical
from matplotlib.pyplot import specgram
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix,)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def noise(data):
    """
    Function to add white noise to our data.

    Parameters
    ----------
        data: np.ndarray
            Data on which tuning is performed.
    Returns
    ----------
    The modified data.
    """

    noise_amp = (
        0.05 * np.random.uniform() * np.amax(data)
    )  # more noise reduce the value to 0.5
    data = data.astype("float64") + noise_amp * np.random.normal(
        size=data.shape[0]
    )
    return data


def speedNpitch(data):
    """
    Function to do speed and Pitch Tuning.

    Parameters
    ----------
        data: np.ndarray
            Data on which tuning is performed.
    Returns
    ----------
    The modified data.
    """

    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.2 / length_change
    tmp = np.interp(
        np.arange(0, len(data), speed_fac),
        np.arange(0, len(data)),
        data
    )
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data


def data_prep(processed_path):
    """Function to prepare data for model creation.

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

    ref = pd.read_csv(processed_path + "/ref.csv")

    df = pd.DataFrame(columns=["feature"])
    df_noise = pd.DataFrame(columns=["feature"])
    df_speedpitch = pd.DataFrame(columns=["feature"])
    cnt = 0

    for i in ref.path:
        X, sample_rate = librosa.load(
            i, res_type="kaiser_fast", duration=2.5, sr=44100, offset=0.5
        )

        mfccs = np.mean(
            librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=13),
            axis=0
        )
        df.loc[cnt] = [mfccs]

        # noise
        aug = noise(X)
        aug = np.mean(
            librosa.feature.mfcc(y=aug, sr=np.array(sample_rate), n_mfcc=13),
            axis=0
        )
        df_noise.loc[cnt] = [aug]

        # speed pitch
        aug = speedNpitch(X)
        aug = np.mean(
            librosa.feature.mfcc(y=aug, sr=np.array(sample_rate), n_mfcc=13),
            axis=0
        )
        df_speedpitch.loc[cnt] = [aug]

        cnt += 1

    # combine the dataframes
    df = pd.concat([ref, pd.DataFrame(df["feature"].values.tolist())], axis=1)
    df_noise = pd.concat(
        [ref, pd.DataFrame(df_noise["feature"].values.tolist())], axis=1
    )
    df_speedpitch = pd.concat(
        [ref, pd.DataFrame(df_speedpitch["feature"].values.tolist())], axis=1
    )
    print(df.shape, df_noise.shape, df_speedpitch.shape)

    df = pd.concat([df, df_noise, df_speedpitch], axis=0, sort=False)
    df = df.fillna(0)

    return df


def split(df):
    """Function to split the Dataframe to training and testing set.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        df: pd.Dataframe
            Feature extracted Dataframe.
        data_folder: StrOrBytesPath
            The path where the data will be saved.
    Returns
    ----------
    X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["path", "labels", "source"], axis=1),
        df.labels,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test):
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


def seq_model_augment(X_train, X_test, y_train, y_test, path):
    """Function to do sequential modelling on augmented data.

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
    The result of modelling performed on augmented data.
    """

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # one hot encode the target
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # New model
    model = Sequential()
    model.add(
        Conv1D(256, 8, padding="same", input_shape=(X_train.shape[1], 1))
    )
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
    model.add(Dense(14))  # Target class number
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
        epochs=2,#100,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    model_name = "seq_model_aug.h5"
    model_path = op.join(path, model_name)
    model.save(model_path)

    model_json = model.to_json()
    with open("augment_model_json.json", "w") as json_file:
        json_file.write(model_json)


def pred(path, processed_path):
    """Function to load the data and model.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        path: StrOrBytesPath
            The path where the artifacts are needs to be saved.
        processed_path: StrOrBytesPAth
            The path where the processed data needs to be saved.
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

    json_file = open("augment_model_json.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(path + "/seq_model_aug.h5")

    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    preds = loaded_model.predict(X_test, batch_size=16, verbose=1)
    preds = preds.argmax(axis=1)

    # predictions
    preds = preds.astype(int).flatten()
    preds = lb.inverse_transform((preds))
    preds = pd.DataFrame({"predictedvalues": preds})

    # Actual labels
    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = lb.inverse_transform((actual))
    actual = pd.DataFrame({"actualvalues": actual})

    # Lets combined both of them into a single dataframe
    finaldf = actual.join(preds)

    # Write out the predictions to disk
    finaldf.to_csv(processed_path + "/Augmodel_predictions.csv", index=False)

    return finaldf


def class_repo(df):
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
        df.actualvalues, df.predictedvalues, target_names=classes
    )


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

    finaldf = pd.read_csv(processed_path + "/Augmodel_predictions.csv")
    modidf = finaldf
    values = {
        "female_angry": "female",
        "female_disgust": "female",
        "female_fear": "female",
        "female_happy": "female",
        "female_sad": "female",
        "female_surprise": "female",
        "female_neutral": "female",
        "male_angry": "male",
        "male_fear": "male",
        "male_happy": "male",
        "male_sad": "male",
        "male_surprise": "male",
        "male_neutral": "male",
        "male_disgust": "male",
    }
    modidf["actualvalues"] = finaldf.actualvalues.replace(values)
    modidf["predictedvalues"] = finaldf.predictedvalues.replace(values)

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

    finaldf = pd.read_csv(processed_path + "/Augmodel_predictions.csv")
    modidf = finaldf
    values = {
        "female_angry": "angry",
        "female_disgust": "disgust",
        "female_fear": "fear",
        "female_happy": "happy",
        "female_sad": "sad",
        "female_surprise": "surprise",
        "female_neutral": "neutral",
        "male_angry": "angry",
        "male_fear": "fear",
        "male_happy": "happy",
        "male_sad": "sad",
        "male_surprise": "surprise",
        "male_neutral": "neutral",
        "male_disgust": "disgust",
    }
    modidf["actualvalues"] = modidf.actualvalues.replace(values)
    modidf["predictedvalues"] = modidf.predictedvalues.replace(values)

    return modidf


# if __name__ == "__main__":
#     from get_argument import argument
#     HERE = op.dirname(op.abspath(__file__))
#     DATA = argument().data
#     SAVE = argument().save
#     PROCESSED_FOLDER = op.join(HERE, DATA, "processed")
#     ARTIFACT_FOLDER = op.join(HERE, SAVE)

#     df = data_prep(processed_path=PROCESSED_FOLDER)
#     X_train, X_test, y_train, y_test = split(df)
#     seq_model_augment(X_train, X_test, y_train, y_test, path=ARTIFACT_FOLDER)

#     final_df = pred(path=ARTIFACT_FOLDER, processed_path=PROCESSED_FOLDER)
#     print(class_repo(final_df))

#     gender_df = gender_test(processed_path=PROCESSED_FOLDER)
#     print(class_repo(gender_df))

#     emotion_df = emotion_test(processed_path=PROCESSED_FOLDER)
#     print(class_repo(emotion_df))
