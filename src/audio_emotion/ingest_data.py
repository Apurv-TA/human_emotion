# -*- coding: utf-8 -*-
"""ingest_data.py docstring

Module created for loading the data needed for the module and to combine them
for easier feature extraction.
"""


import os
import os.path as op

import pandas as pd


def load_data_SAVEE(SAVEE):
    """Function to read and load SAVEE dataset.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        SAVEE: StrOrBytesPath
            The path where the SAVEE data is downloaded.
    Returns
    ----------
    A dataframe having emotion of the speaker categorized along with the
    file path.
    """

    dir_list = sorted(os.listdir(SAVEE))

    emotion = []
    path = []
    for i in dir_list:
        if i[-8:-6] == "_a":
            emotion.append("male_angry")
        elif i[-8:-6] == "_d":
            emotion.append("male_disgust")
        elif i[-8:-6] == "_f":
            emotion.append("male_fear")
        elif i[-8:-6] == "_h":
            emotion.append("male_happy")
        elif i[-8:-6] == "_n":
            emotion.append("male_neutral")
        elif i[-8:-6] == "sa":
            emotion.append("male_sad")
        elif i[-8:-6] == "su":
            emotion.append("male_surprise")
        else:
            emotion.append("male_error")
        path.append(SAVEE + "/" + i)

    SAVEE_df = pd.DataFrame(emotion, columns=["labels"])
    SAVEE_df["source"] = "SAVEE"
    SAVEE_df = pd.concat(
        [
            SAVEE_df, pd.DataFrame(path, columns=["path"])
        ],
        axis=1
    )

    return SAVEE_df


def load_data_RAVDESS(RAVDESS):
    """Function to read and load RAVDESS dataset.

    The default arguments of the function can be overwritten when supplied by
    the user.

    Parameters
    ----------
        RAVDESS: StrOrBytesPath
            The path where the RAVDESS data is downloaded.
    Returns
    ----------
    A dataframe having emotion of the speaker categorized along with the
    file path.
    """

    dir_list = sorted(os.listdir(RAVDESS))

    emotion = []
    gender = []
    path = []
    for i in dir_list:
        fname = os.listdir(RAVDESS + "/" + i)
        for f in fname:
            part = f.split(".")[0].split("-")
            emotion.append(int(part[2]))
            temp = int(part[6])
            if temp % 2 == 0:
                temp = "female"
            else:
                temp = "male"
            gender.append(temp)
            path.append(RAVDESS + "/" + i + "/" + f)

    RAVDESS_df = pd.DataFrame(emotion)
    RAVDESS_df = RAVDESS_df.replace(
        {
            1: "neutral",
            2: "neutral",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fear",
            7: "disgust",
            8: "surprise",
        }
    )
    RAVDESS_df = pd.concat([pd.DataFrame(gender), RAVDESS_df], axis=1)
    RAVDESS_df.columns = ["gender", "emotion"]
    RAVDESS_df["labels"] = RAVDESS_df.gender + "_" + RAVDESS_df.emotion
    RAVDESS_df["source"] = "RAVDESS"
    RAVDESS_df = pd.concat(
        [
            RAVDESS_df, pd.DataFrame(path, columns=["path"])
        ],
        axis=1
    )
    RAVDESS_df = RAVDESS_df.drop(["gender", "emotion"], axis=1)

    return RAVDESS_df


def load_data_TESS(TESS):
    """Function to read and load TESS dataset.

    The default arguments of the function can be overwritten when supplied by
    the user.

    Parameters
    ----------
        TESS: StrOrBytesPath
            The path where the TESS data is downloaded.
    Returns
    ----------
    A dataframe having emotion of the speaker categorized along with the
    file path.
    """

    dir_list = sorted(os.listdir(TESS))
    path = []
    emotion = []

    for i in dir_list:
        fname = os.listdir(TESS + "/" + i)
        for f in fname:
            if i == "OAF_angry" or i == "YAF_angry":
                emotion.append("female_angry")
            elif i == "OAF_disgust" or i == "YAF_disgust":
                emotion.append("female_disgust")
            elif i == "OAF_Fear" or i == "YAF_fear":
                emotion.append("female_fear")
            elif i == "OAF_happy" or i == "YAF_happy":
                emotion.append("female_happy")
            elif i == "OAF_neutral" or i == "YAF_neutral":
                emotion.append("female_neutral")
            elif (
                i == "OAF_Pleasant_surprise" or i == "YAF_pleasant_surprised"
            ):
                emotion.append("female_surprise")
            elif i == "OAF_Sad" or i == "YAF_sad":
                emotion.append("female_sad")
            else:
                emotion.append("Unknown")
            path.append(TESS + "/" + i + "/" + f)

    TESS_df = pd.DataFrame(emotion, columns=["labels"])
    TESS_df["source"] = "TESS"
    TESS_df = pd.concat(
        [
            TESS_df, pd.DataFrame(path, columns=["path"])
        ],
        axis=1
    )

    return TESS_df


def load_data_CREMA(CREMA):
    """Function to read and load CREMA dataset.

    The default arguments of the function can be overwritten when supplied by
    the user.

    Parameters
    ----------
        CREMA: StrOrBytesPath
            The path where the CREMA data is downloaded.
    Returns
    ----------
    A dataframe having emotion of the speaker categorized along with the
    file path.
    """

    dir_list = sorted(os.listdir(CREMA))

    gender = []
    emotion = []
    path = []
    female = [
        1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018,
        1020, 1021, 1024, 1025, 1028, 1029, 1030, 1037, 1043, 1046, 1047,
        1049, 1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072,
        1073, 1074, 1075, 1076, 1078, 1079, 1082, 1084, 1089, 1091
    ]

    for i in dir_list:
        part = i.split("_")
        if int(part[0]) in female:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        if part[2] == "SAD" and temp == "male":
            emotion.append("male_sad")
        elif part[2] == "ANG" and temp == "male":
            emotion.append("male_angry")
        elif part[2] == "DIS" and temp == "male":
            emotion.append("male_disgust")
        elif part[2] == "FEA" and temp == "male":
            emotion.append("male_fear")
        elif part[2] == "HAP" and temp == "male":
            emotion.append("male_happy")
        elif part[2] == "NEU" and temp == "male":
            emotion.append("male_neutral")
        elif part[2] == "SAD" and temp == "female":
            emotion.append("female_sad")
        elif part[2] == "ANG" and temp == "female":
            emotion.append("female_angry")
        elif part[2] == "DIS" and temp == "female":
            emotion.append("female_disgust")
        elif part[2] == "FEA" and temp == "female":
            emotion.append("female_fear")
        elif part[2] == "HAP" and temp == "female":
            emotion.append("female_happy")
        elif part[2] == "NEU" and temp == "female":
            emotion.append("female_neutral")
        else:
            emotion.append("Unknown")
        path.append(CREMA + "/" + i)

    CREMA_df = pd.DataFrame(emotion, columns=["labels"])
    CREMA_df["source"] = "CREMA"
    CREMA_df = pd.concat(
        [
            CREMA_df, pd.DataFrame(path, columns=["path"])
        ],
        axis=1
    )

    return CREMA_df


def load_data(data_path):
    """Function to combine all the loading functions present in this module.
    The functions are 'load_data_SAVEE', 'load_data_RAVDESS',
    'load_data_TESS' and 'load_data_CREMA'.

    Additionally it also creates and saves a reference file containing the
    emotion and the path of the file.

    The default arguments of the function can be overwritten when supplied
    by the user.

    Parameters
    ----------
        data_path: StrOrBytesPath
            The location where data is to stored(downloaded).
    Returns
    ----------
    A dataframe having emotion of the speaker categorized along with the
    file path.
    """
    SAVEE_df = load_data_SAVEE(SAVEE=op.join(data_path, "raw", "SAVEE"))
    RAVDESS_df = load_data_RAVDESS(
        RAVDESS=op.join(data_path, "raw", "RAVDESS")
    )
    TESS_df = load_data_TESS(TESS=op.join(data_path, "raw", "TESS"))
    CREMA_df = load_data_CREMA(CREMA=op.join(data_path, "raw", "CREMA"))

    ref = pd.concat([SAVEE_df, RAVDESS_df, TESS_df, CREMA_df], axis=0)

    ref.to_csv(op.join(data_path, "processed") + "/ref.csv", index=False)

    return ref
