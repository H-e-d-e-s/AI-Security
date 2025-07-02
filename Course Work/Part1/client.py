from utils.cl_utils import timeit
from sklearn.ensemble import RandomForestClassifier

import sys
import json

import pandas as pd
import requests

UID = 2607
COMMANDS = ["-a", "-e", "-h", "-m", "-r", "-t"]
example_input = [
    {"UID": UID},  # This must be provided at index 0
    {
        "LIMIT_BAL": 1,
        "AGE": 2,
        "PAY_0": 3,
        "PAY_2": 4,
        "PAY_3": 5,
        "PAY_4": 6,
        "PAY_5": 7,
        "PAY_6": 8,
        "BILL_AMT1": 9,
        "BILL_AMT2": 10,
        "BILL_AMT3": 11,
        "BILL_AMT4": 12,
        "BILL_AMT5": 13,
        "BILL_AMT6": 14,
        "PAY_AMT1": 15,
        "PAY_AMT2": 16,
        "PAY_AMT3": 17,
        "PAY_AMT4": 18,
        "PAY_AMT5": 19,
        "PAY_AMT6": 20,
        "default payment next month": 0
    }
]
multiple_example_inputs = [
    {"UID": UID},
    {"LIMIT_BAL": 1,"AGE": 2,"PAY_0": 3,"PAY_2": 4,"PAY_3": 5,"PAY_4": 6,"PAY_5": 7,"PAY_6": 8,"BILL_AMT1": 9,"BILL_AMT2": 10,"BILL_AMT3": 11,"BILL_AMT4": 12,"BILL_AMT5": 13,"BILL_AMT6": 14,"PAY_AMT1": 15,"PAY_AMT2": 16,"PAY_AMT3": 17,"PAY_AMT4": 18,"PAY_AMT5": 19,"PAY_AMT6": 20,"default payment next month": 0},
    {"LIMIT_BAL": 1,"AGE": 2,"PAY_0": 3,"PAY_2": 4,"PAY_3": 5,"PAY_4": 6,"PAY_5": 7,"PAY_6": 8,"BILL_AMT1": 9,"BILL_AMT2": 10,"BILL_AMT3": 11,"BILL_AMT4": 12,"BILL_AMT5": 13,"BILL_AMT6": 14,"PAY_AMT1": 15,"PAY_AMT2": 16,"PAY_AMT3": 17,"PAY_AMT4": 18,"PAY_AMT5": 19,"PAY_AMT6": 20,"default payment next month": 1}
]    

PATHS = {
    "original_data": "data/data.csv",
    "train_data": "data/train_data.csv",
    "test_data": "data/test_data.csv",
    "original_model": "models/rfc_original.pkl",
    "latest_model": "models/rfc_latest.pkl",
    "model_metadata": "models/model_metadata.ini"
}

def launch_attack():  # Implement this
    """ python client.py -a
    Launches a data poisoning attack on the server by feeding it with
    incorrectly labeled data """

    # Read legitimate data
    train_data = pd.read_csv(PATHS["train_data"])

    # Example of code that changes the first feature in the data to integer 
    # value 42 looks as follows: train_data["LIMIT_BAL"] = 42
    # Add your own attack code below.
    
    # <START ATTACK CODE>
    x_train=train_data.drop(['default payment next month'],axis=1)
    y_train=train_data['default payment next month']
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_train)
    for i in range(len(y_pred)):
        if y_pred[i]==0:
            y_pred[i]=1
        else:
            y_pred[i]=0

    for i in range(0,10000):
        train_data['default payment next month']=y_pred

    # <END ATTACK CODE, leave code below unchanged>
    

    payload = train_data.to_dict("records")
    payload.insert(0, {"UID": UID})

    # Launch the attack
    print("Launching attack, please wait...")
    submit_data(payload)


@timeit
def submit_data(inp):
    """ python client.py -d
    Requests the server to store the submitted data for training of the
    server's ML-based classification system

    The input (inp) must be a list of dictionaries containing all the keys 
    and with all dict values being integers, see example entry above. The 
    class attribute 'default payment next month' must be either 0 or 1. 
    The first element of the list must be a dict with the single key 'UID'.

    For bulk submission, just append more dictionaries to the list. """

    r = requests.post("http://localhost:5000/add", json=inp)

    if r.status_code == 200:
        result = json.loads(r.text)
        print(f"Successful additions: {result['successful']}")
        print(f"Failed additions: {result['failed']}")
    else:
        print(r.status_code, r.reason)


def get_model_report():
    """ python client.py -m
    Requests a classification performance report of the trained model on the
    server and prints the results """

    r = requests.get("http://localhost:5000/report")

    if r.status_code == 200:
        result = json.loads(r.text)

        print("Model classification performance report:")
        print(f"  Algorithm:    {result['alg']}")
        print(f"  Last trained: {result['date']}")
        print(f"  Trained on:   {result['size']} entries")
        print(f"  Accuracy:     {round(result['acc'] * 100, 2)}%")
        print(f"  Precision:    {round(result['pre'] * 100, 2)}%")
        print(f"  Recall:       {round(result['rec'] * 100, 2)}%")
        print(f"  F1-score:     {round(result['f1s'] * 100, 2)}%")
        print(f"  ROC-AUC:      {round(result['auc'] * 100, 2)}%")
    else:
        print(r.status_code, r.reason)


def print_usage():
    """ python client.py -h
    Prints the available commands for the client application """

    print("Available commands:")
    print("   -a      Launch a data poisoning attack on the server.")
    print("   -e      Example of new data submission for storage in the "
          "server's dataset.")
    print("   -h      Print this help message.")
    print("   -m      Retrieve a classification performance report of the "
          "server's trained model.")
    print("   -r      Reset the state of the machine learning model and the "
          "stored data.")
    print("   -t      Start training the model on all collected data.")


@timeit
def reset_state():
    """ python client.py -r
    Resets the machine learning model and it's stored data on the server. """
    
    #answer = input("Are you sure you want to reset the server's machine "
    #               "learning model and it's stored data? [y/N] ")
    #if answer != 'y':
    #    print("Aborting!")
    #    return

    r = requests.post("http://localhost:5000/reset")

    if r.status_code == 200:
        print(r.text)
    else:
        print(r.status_code, r.reason)


@timeit
def train_model():
    """ python client.py -t
    Requests the server to train the model with the available training data """

    print("Training model, please wait...")
    r = requests.post("http://localhost:5000/train")

    if r.status_code == 200:
        print(r.text)
    else:
        print(r.status_code, r.reason)


if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    print_usage()
    exit()
elif sys.argv[1] == '-a':
    launch_attack()
elif sys.argv[1] == '-e':
    #submit_data(example_input)
    submit_data(multiple_example_inputs)
elif sys.argv[1] == '-h':
    print_usage()
elif sys.argv[1] == '-m':
    get_model_report()
elif sys.argv[1] == '-r':
    reset_state()
elif sys.argv[1] == '-t':
    train_model()
