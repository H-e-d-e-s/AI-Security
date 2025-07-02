import warnings
import configparser
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PATHS = {
    "original_data": "data/data.csv",
    "train_data": "data/train_data.csv",
    "test_data": "data/test_data.csv",
    "original_model": "models/rfc_original.pkl",
    "latest_model": "models/rfc_latest.pkl",
    "model_metadata": "models/model_metadata.ini"
}

duplicates = 0

def get_dataset():
    """"""
    return pd.read_csv(PATHS["original_data"])


def get_model_report():
    """"""
    X_test, y_test = get_test_data()

    try:
        with open(PATHS["latest_model"], "rb") as model_file:
            clf = pickle.load(model_file)
    except FileNotFoundError:
        with open(PATHS["original_model"], "rb") as model_file:
            clf = pickle.load(model_file)
    y_pred = clf.predict(X_test)

    config = configparser.ConfigParser()
    config.read(PATHS["model_metadata"])

    return {
        "alg": config["latest_model"]["algorithm"],
        "date": config["latest_model"]["datetime"],
        "size": config["latest_model"]["train_size"],
        "acc": accuracy_score(y_test, y_pred),
        "pre": precision_score(y_test, y_pred, average='weighted'),
        "rec": recall_score(y_test, y_pred, average='weighted'),
        "f1s": f1_score(y_test, y_pred, average='weighted'),
        "auc": roc_auc_score(y_test, y_pred, average='weighted')
    }


def _generate_train_test_csv():
    """"""
    data = get_dataset()

    X = data.drop(columns=["default payment next month"])
    y = data["default payment next month"].astype('int')

    train, test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, random_state=42)

    train["default payment next month"] = y_train
    test["default payment next month"] = y_test

    train.to_csv(PATHS["train_data"], index=False)
    test.to_csv(PATHS["test_data"], index=False)


def _get_partial_data(part, x_y_split):
    if part == "train":
        data = pd.read_csv(PATHS["train_data"])
    elif part == "test":
        data = pd.read_csv(PATHS["test_data"])
    else:
        raise ValueError(f"Invalid part value: {part}")

    if not x_y_split:
        return data

    y = data["default payment next month"]
    X = data.drop(columns=["default payment next month"])

    return X, y


def get_test_data(x_y_split=True):
    """"""
    return _get_partial_data("test", x_y_split)


def get_train_data(x_y_split=True):
    """"""
    return _get_partial_data("train", x_y_split)


def reset_model_and_data():
    """"""
    _generate_train_test_csv()
    train_and_save_model(reset=True)
    try:
        os.remove(PATHS["latest_model"])
    except FileNotFoundError:
        print("No old model to delete.")


def store_data(submitted_data):
    """"""
    # Check that an UID was provided in the submitted data
    global duplicates

    if "UID" not in submitted_data[0]:
        print("Missing UID in submitted data!")
        return len(submitted_data)

    failed_entries = 0
    train_data = get_train_data(x_y_split=False)

    print( "Added, but duplicate entry(s) existed in the data:", end="", flush=True)
    duplicates = 0
    for entry in submitted_data[1:]:
        try:
            validated_entry = _validate_input(entry, train_data)
            train_data = pd.concat([train_data, pd.DataFrame.from_records([validated_entry]) ] )
        except Exception as e:
            failed_entries += 1
            print(e)

    count = len(submitted_data)
    print("| %d/%d (%.2f %%)" % (duplicates, count, (duplicates/count)*100 ) )
    train_data.to_csv(PATHS["train_data"], index=False)

    return failed_entries


def train_and_save_model(reset=False):
    """"""
    X_train, y_train = get_train_data()
    random_state = None
    save_path = PATHS["latest_model"]

    if reset:
        random_state = 42
        save_path = PATHS["original_model"]

    algorithm = "Random Forest"
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    with open(save_path, "wb") as model_file:
        pickle.dump(clf, model_file)

    config = configparser.ConfigParser()
    config["latest_model"] = {}
    config["latest_model"]["algorithm"] = algorithm
    config["latest_model"]["datetime"] = \
        datetime.now().strftime('%Y-%m-%d %H:%M')
    config["latest_model"]["train_size"] = str(X_train.shape[0])

    with open(PATHS["model_metadata"], 'w+') as config_file:
        config.write(config_file)


def _validate_input(inp, train_data):
    """"""
    # Check that all required features exist in input
    global duplicates

    for column in train_data.columns:
        if column not in inp.keys():
            raise ValueError(f"Missing input attribute {column}.")

    # Check that all input attributes are part of the stored data
    for attribute in inp.keys():
        if attribute not in train_data.columns:
            raise ValueError(f"Unexpected attribute {attribute} in input.")

    # Check that amount of input attributes match the stored data
    if len(inp) != len(train_data.columns):
        raise ValueError("Number of input attributes don't match the stored "
                         "data.")

    # Check that all input values are numeric
    for value in inp.values():
        try:
            int(value)
        except ValueError:
            raise ValueError(f"Attribute value '{value}' is not numeric.")

    # Check that label is binary (0 or 1)
    if inp["default payment next month"] not in [0, 1]:
        raise ValueError(f"Class attribute not '0' or '1'.")

    if (train_data == np.array(list(inp.values())).astype("int")).all(1).any():
        #raise ValueError("Entry already exists in exeisting data.")
        duplicates += 1
        print("*", end="", flush=True)

    return inp
