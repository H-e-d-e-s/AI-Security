from utils.ml_utils import get_model_report, reset_model_and_data, store_data, train_and_save_model
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def index():
    return 'Server is running!'


@app.route('/report')
def model_report():
    response = {}
    try:
        response = get_model_report()
    except Exception as e:
        print(e)

    return response


@app.route('/add', methods=['POST'])
def add_data():
    response = {"successful": 0, "failed": 0}
    try:
        data_entries = request.get_json()
        failed_entries = store_data(data_entries)
        added_entries = len(data_entries) - 1 - failed_entries
        response["successful"] = added_entries
        response["failed"] = failed_entries
    except Exception as e:
        print(e)

    return response


@app.route('/reset', methods=['POST'])
def reset():
    response = "Failed to reset model and/or data.."

    try:
        reset_model_and_data()
        response = "Model and data has been successfully reset!"
    except Exception as e:
        print(e)

    return response


@app.route('/train', methods=['POST'])
def train_model():
    response = "Model training failed.."

    try:
        train_and_save_model()
        response = "Model has successfully been trained!"
    except Exception as e:
        print(e)

    return response

if __name__ == "__main__":
    app.run()
