# -*- coding: utf-8 -*-
"""feature_extraction.py docstring

Module created for extracting and converting the feature from the data
to a form which is needed for model creation. 

Additionally it also splits the data to training and testing set.
"""


import os
import os.path as op
import warnings

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore")
np.random.seed(42)


def feature_extraction(ref, data_folder):
    """Function to extract feature from reference data. It also saves the
    extracted data for future use.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        ref: pd.Dataframe
            Reference file containing the path to the audio files as well as
            the emotion associated with each audio.
        data_folder: StrOrBytesPath
            The path where the processed data needs to be saved.
    Returns
    ----------
    A dataframe having emotion of the speaker categorized along with the
    features.
    """

    df = pd.DataFrame(columns=['feature'])

    counter = 0
    for index, path in enumerate(ref.path):
        X, sample_rate = librosa.load(
            path,
            res_type='kaiser_fast',
            duration=2.5,
            sr=44100,
            offset=0.5
        )
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(
            librosa.feature.mfcc(
                y=X,
                sr=sample_rate,
                n_mfcc=13
            ),
            axis=0
        )
        df.loc[counter] = [mfccs]
        counter = counter + 1

    ref.reset_index(inplace=True, drop=True)
    df = pd.concat(
        [
            ref,
            pd.DataFrame(df['feature'].values.tolist())
        ],
        axis=1
    )
    df = df.fillna(0)
    df.to_csv(data_folder + "/data.csv", index=False)

    return df


def df_split(df, data_folder):
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
    Training and testing data.
    """

    train, test = train_test_split(
        df.drop(
            ['path', 'source'],
            axis=1
        ),
        test_size=0.25,
        shuffle=True,
        random_state=42
    )

    train.to_csv(data_folder + "/train.csv", index=False)
    test.to_csv(data_folder + "/test.csv", index=False)

    return train, test


def extract_split(ref, data_folder):
    """Function to combine all the functions in this module.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        ref: pd.Dataframe
            Reference file containing the path to the audio files as well as
            the emotion associated with each audio.
        data_folder: StrOrBytesPath
            The path where the processed data needs to be saved.
    Returns
    ----------
    Feature extracted dataframe and training and testing dataframes.
    """

    df = feature_extraction(ref=ref, data_folder=data_folder)
    train, test = df_split(df, data_folder=data_folder)

    return df, train, test

# if __name__ == "__main__":
#     from get_argument import argument

#     HERE = op.dirname(op.abspath(__file__))
#     DATA = argument().data
#     PROCESSED_FOLDER = op.join(HERE, DATA, "processed")

#     ref = pd.read_csv(PROCESSED_FOLDER + "/ref.csv")
#     df, train, test = extract_split(
#         ref=ref,
#         data_folder=PROCESSED_FOLDER
#     )
