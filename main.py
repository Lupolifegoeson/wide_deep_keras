#!/usr/bin/env python
# encoding: utf-8

import subprocess
import numpy as np
import pandas as pd
import os
import fire
from keras.layers import Input, Embedding
from keras.layers import Flatten, Dense, concatenate, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1_l2
from sklearn.preprocessing import Imputer, PolynomialFeatures, LabelEncoder


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


def _embedding_input(n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64')
    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)


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

    categorical_columns = ["education", "workclass", "marital_status", "occupation", "income_bracket"]
    TARGET = "income_bracket"
    # target_column = "income_bracket"


    train_data = train_data[categorical_columns]
    test_data = test_data[categorical_columns]

    train_data["IS_TRAIN"] = 1
    test_data["IS_TRAIN"] = 0

    df_all = pd.concat([train_data, test_data], ignore_index=True)

    df_all["income_bracket"] = df_all["income_bracket"].apply(lambda x: _clean_target(x))

    # print(df_all["income_bracket"].value_counts())

    le = LabelEncoder()
    le_count = 0

    for col in categorical_columns:
        le.fit(df_all[col])
        df_all[col] = le.transform(df_all[col])
        le_count += 1


    train_x_df = df_all.loc[df_all["IS_TRAIN"] == 1].drop(columns=["IS_TRAIN", TARGET], axis=1)
    train_x = [train_x_df[_] for _ in list(train_x_df.columns)]
    train_y = np.array(df_all.loc[df_all["IS_TRAIN"] == 1][TARGET].values).reshape(-1, 1)
    test_x_df = df_all.loc[df_all["IS_TRAIN"] == 0].drop(["IS_TRAIN", TARGET], axis=1)
    test_x = [test_x_df[_] for _ in list(test_x_df.columns)]
    test_y = np.array(df_all.loc[df_all["IS_TRAIN"] == 0][TARGET].values).reshape(-1, 1)

    return train_x, train_y, test_x, test_y

def model_deep():
    """deep model building"""
    MODEL_SETTING = {
        "DIM": 10,
        "REG": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 10
    }

    data = prepare_data()

    embedding_cols = list(data.columns)
    embedding_cols.remove("IS_TRAIN")
    embedding_cols.remove(TARGET)

    embedding_tensors = []

    for _ in embedding_cols:
        number_input = data[_].nunique()
        tensor_input, tensor_build = _embedding_input(
            number_input, MODEL_SETTING["DIM"], MODEL_SETTING["REG"]
        )
        embedding_tensors.append((tensor_input, tensor_build))

    input_layer = [_[0] for _ in embedding_tensors]
    input_embed = [_[1] for _ in embedding_tensors]

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


if __name__ == "__main__":
    fire.Fire({
        "prepare": prepare_data,
        "deep": model_deep
    })
