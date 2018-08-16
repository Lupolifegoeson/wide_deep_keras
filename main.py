#!/usr/bin/env python
# encoding: utf-8

import subprocess
import numpy as np
import pandas as pd
import os
import fire
from keras.layers import Input, Embedding
from keras.layers import Flatten, Dense, concatenate, Dropout, Reshape
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1_l2
from sklearn.preprocessing import Imputer, PolynomialFeatures, LabelEncoder, MinMaxScaler


def _download():
    """
    Download data from ics uci
    """
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_bracket']

    train_data = "train.csv"
    train_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_data = "test.csv"
    test_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    data_src = {train_data: train_data_url,
                test_data: test_data_url}

    for data_file, data_url in data_src.items():
        if not os.path.exists(data_file):
            print("Downloading {} ============".format(data_file))
            cmd = "wget -O {0} {1}".format(data_file, data_url)
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
        else:
            print("{} found so we skip ==========".format(data_file))

    train_data_df = pd.read_csv(train_data,
                                names=COLUMNS,
                                skipinitialspace=True)

    test_data_df = pd.read_csv(test_data,
                               names=COLUMNS,
                               skipinitialspace=True,
                               skiprows=1)

    return train_data_df, test_data_df


def _categorical_input(n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64')
    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)


def _numerical_input():
    inp = Input(shape=(1,), dtype='float32')
    return inp, Reshape((1, 1))(inp)


def _clean_target(x):
    if x[-1] == ".":
        return x[:-1]
    else:
        return x


def prepare_data():
    """
    Prepare data for training
    1. Label encoding for categorical columns
    """
    train_data, test_data = _download()

    CAT_FEATURES = ["education", "workclass", "marital_status",
                    "occupation", "relationship", "gender", "native_country",
                    "race"]

    NUM_FEATURES = ["age", "hours_per_week", "capital_gain", "capital_loss"]
    # NUM_FEATURES = []

    TARGET = "income_bracket"

    columns_used = CAT_FEATURES+NUM_FEATURES
    columns_used.append(TARGET)

    TRAIN_FLAG = "IS_TRAIN"

    train_data = train_data[columns_used]
    test_data = test_data[columns_used]

    train_data[TRAIN_FLAG] = 1
    test_data[TRAIN_FLAG] = 0

    df_all = pd.concat([train_data, test_data], ignore_index=True)

    df_all["income_bracket"] = df_all["income_bracket"].apply(lambda x: _clean_target(x))

    le = LabelEncoder()
    le_count = 0

    for col in CAT_FEATURES:
        le.fit(df_all[col])
        df_all[col] = le.transform(df_all[col])
        le_count += 1

    le.fit(df_all[TARGET])
    df_all[TARGET] = le.transform(df_all[TARGET])
    le_count += 1

    return df_all, TRAIN_FLAG, TARGET, CAT_FEATURES, NUM_FEATURES


def deep_model():
    """deep model building"""
    MODEL_SETTING = {
        "DIM": 5,
        "REG": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 10
    }

    data, TRAIN_FLAG, TARGET, CAT_FEATURES, NUM_FEATURES = prepare_data()

    train_x_df = data.loc[data[TRAIN_FLAG] == 1].drop(columns=[TRAIN_FLAG, TARGET], axis=1)
    train_x = [train_x_df[_] for _ in list(train_x_df.columns)]
    train_y = np.array(data.loc[data[TRAIN_FLAG] == 1][TARGET].values).reshape(-1, 1)
    test_x_df = data.loc[data[TRAIN_FLAG] == 0].drop([TRAIN_FLAG, TARGET], axis=1)
    test_x = [test_x_df[_] for _ in list(test_x_df.columns)]
    test_y = np.array(data.loc[data[TRAIN_FLAG] == 0][TARGET].values).reshape(-1, 1)

    print(train_x[0])


    embedding_tensors = []
    for _ in CAT_FEATURES:
        number_input = data[_].nunique()
        tensor_input, tensor_build = _categorical_input(
            number_input, MODEL_SETTING["DIM"], MODEL_SETTING["REG"]
        )
        embedding_tensors.append((tensor_input, tensor_build))

    continuous_tensors = []
    for _ in  NUM_FEATURES:
        tensor_input, tensor_build = _numerical_input()
        continuous_tensors.append((tensor_input, tensor_build))

    input_layer = [_[0] for _ in embedding_tensors]
    input_layer += [_[0] for _ in continuous_tensors]

    input_embed = [_[1] for _ in embedding_tensors]
    input_embed += [_[1] for _ in continuous_tensors]

    x = concatenate(input_embed, axis=-1)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(1, activation="sigmoid")(x)

    deep_model = Model(input_layer, x)

    deep_model.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["acc"])

    deep_model.fit(train_x, train_y,
                   batch_size=MODEL_SETTING["BATCH_SIZE"],
                   epochs=MODEL_SETTING["EPOCHS"],
                   validation_data=(test_x, test_y))

    eval_res = deep_model.evaluate(test_x, test_y)

    print("eval results: ", eval_res)


def wide_model():
    MODEL_SETTING = {
        "DIM": 5,
        "REG": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 20}

    data, TRAIN_FLAG, TARGET, CAT_FEATURES, NUM_FEATURES = prepare_data()

    data = pd.get_dummies(data, columns=[_ for _ in CAT_FEATURES])

    train_x_df = data.loc[data[TRAIN_FLAG] == 1].drop(columns=[TRAIN_FLAG, TARGET], axis=1)
    # train_x = [train_x_df[_] for _ in list(train_x_df.columns)]
    train_x = train_x_df.values

    train_y = np.array(data.loc[data[TRAIN_FLAG] == 1][TARGET].values).reshape(-1, 1)
    test_x_df = data.loc[data[TRAIN_FLAG] == 0].drop([TRAIN_FLAG, TARGET], axis=1)
    # test_x = [test_x_df[_] for _ in list(test_x_df.columns)]
    test_x = test_x_df.values

    test_y = np.array(data.loc[data[TRAIN_FLAG] == 0][TARGET].values).reshape(-1, 1)

    scaler = MinMaxScaler()

    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)

    input_layer = Input(shape=(train_x.shape[1],), dtype='float32')
    x = Dense(train_y.shape[1], activation="relu")(input_layer)
    wide_model = Model(input_layer, x)

    wide_model.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["acc"])

    wide_model.fit(train_x, train_y,
                   epochs=MODEL_SETTING["EPOCHS"],
                   batch_size=MODEL_SETTING["BATCH_SIZE"],
                   validation_data=(test_x, test_y))

    results = wide_model.evaluate(test_x, test_y)
    print("\n", results)


if __name__ == "__main__":
    fire.Fire({
        "prepare": prepare_data,
        "deep": deep_model,
        "wide": wide_model
    })
