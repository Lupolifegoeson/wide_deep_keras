#!/usr/bin/env python
# encoding: utf-8

import subprocess
import numpy as np
import pandas as pd
import os
import fire

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


def prepare_data():
    """
    Prepare data for training
    """

    train_data, test_data = _download()
    print(train_data.head(5))
    print(test_data.head(5))



if __name__ == "__main__":
    fire.Fire({
        "prepare": prepare_data

    })
