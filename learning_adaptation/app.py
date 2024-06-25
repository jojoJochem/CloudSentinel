from flask import Flask, request, jsonify, Response
import pandas as pd
import numpy as np
import json
import logging
import traceback
import requests
import torch
import shutil
from celery import Celery
import os
from flask_cors import CORS

from cgnn.config import set_config, get_config, set_initial_config
from tasks import train_and_evaluate_task

# Initialize Flask app and configure CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Celery
app.config.update(
    CELERY_BROKER_URL='redis://redis:6379/2',
    CELERY_RESULT_BACKEND='redis://redis:6379/2'
)
celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'], broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/cgnn_train_model', methods=['POST'])
def cgnn_train_models():
    """
    Handles the training request for CGNN models.
    """
    try:
        train_files = request.files
        train_info_json = request.form.get('train_info')
        train_info = json.loads(train_info_json)

        train_array = pd.read_csv(train_files['train_array'], header=None).to_numpy(dtype=np.float32).tolist()
        test_array = pd.read_csv(train_files['test_array'], header=None).to_numpy(dtype=np.float32).tolist()
        anomaly_label_array = pd.read_csv(train_files['anomaly_label_array'], header=None).to_numpy(dtype=np.float32).tolist()

        logger.info("Received training request.")
        logger.debug(f"Train Info: {train_info}")

        task = train_and_evaluate_task.apply_async(args=[train_array, test_array, anomaly_label_array, train_info])

        return jsonify({"task_id": task.id}), 202
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/get_status/<task_id>', methods=['GET'])
def get_status(task_id):
    """
    Retrieves the status of a Celery task.

    Args:
        task_id (str): The ID of the task.

    Returns:
        json: The task status and info.
    """
    task = train_and_evaluate_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending... (this may take a while)'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return jsonify(response)


@app.route('/get_available_models', methods=['GET'])
def get_available_models():
    """
    Retrieves the available trained models.

    Returns:
        json: A dictionary of models with their parameters, configuration, and evaluation.
    """
    model_dir = 'trained_models_temp'
    models = {}
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        with open(os.path.join(model_path, 'model_params.json'), 'r') as f:
            params = json.load(f)
        with open(os.path.join(model_path, 'model_config.json'), 'r') as f:
            config = json.load(f)
        with open(os.path.join(model_path, 'model_evaluation.json'), 'r') as f:
            evaluation = json.load(f)
        models[model_name] = {'model_params': params, 'model_config': config, 'model_evaluation': evaluation}

    return jsonify(models)


@app.route('/save_to_detection_module', methods=['POST'])
def save_to_detection_module():
    """
    Saves a trained model to the detection module.

    Returns:
        json: Success message or error details.
    """
    model_info_json = request.form.get('model_info')
    model_info = json.loads(model_info_json)
    path = f"trained_models_temp/{next(iter(model_info['data']))}"
    model = torch.load(path+'/model.pt', map_location='cpu')
    json_data = {key: value.tolist() for key, value in model.items()}
    model_json = json.dumps(json_data, indent=2)

    try:
        model_info_json = json.dumps(model_info)
        response = requests.post(model_info['settings']['API_CGNN_ANOMALY_DETECTION_URL'] + '/save_model',
                                 files={'model': model_json}, data={'model_info': model_info_json})

        if response.status_code == 200:
            shutil.rmtree(path)
        return jsonify("success"), 200
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/cgnn_train_with_existing_dataset', methods=['POST'])
def cgnn_train_with_existing_dataset():
    """
    Trains CGNN with an existing dataset.

    Returns:
        Response: The response from the data processing service.
    """
    train_info_json = request.form.get('train_info')
    train_info = json.loads(train_info_json)

    dataset = train_info['data']['dataset']
    directory = 'datasets/'+dataset
    test_label_filename = f"test_label_{dataset}"
    test_filename = f"test_{dataset}"
    train_filename = f"train_{dataset}"

    for filename in os.listdir(directory):
        if filename == test_label_filename:
            anomaly_label_array = pd.read_csv(directory+'/'+test_label_filename, header=None)
        elif filename == test_filename:
            test_array = pd.read_csv(directory+'/'+test_filename, header=None)
        elif filename == train_filename:
            train_array = pd.read_csv(directory+'/'+train_filename, header=None)

    # read the details.json from the directory
    details_file = os.path.join(directory, 'details.json')
    with open(details_file, 'r') as file:
        model_config = json.load(file)

    dataset_containers = model_config['containers']
    dataset_metrics = model_config['metrics']

    # Generate new headers for all combinations of containers and metrics
    new_headers = [f"{container}_{metric}" for container in dataset_containers for metric in dataset_metrics]
    # Rename the dataframe columns
    train_array.columns = new_headers
    test_array.columns = new_headers

    selected_containers = train_info['data']['containers']
    selected_metrics = train_info['data']['metrics']

    selected_headers = [f"{container}_{metric}" for container in selected_containers for metric in selected_metrics]
    train_array_filtered = train_array[selected_headers]
    test_array_filtered = test_array[selected_headers]

    train_files = {
        'train_array': train_array_filtered.to_csv(header=False, index=False),
        'test_array': test_array_filtered.to_csv(header=False, index=False),
        'anomaly_label_array': anomaly_label_array.to_csv(header=False, index=False)
    }
    train_info['data']['step_size'] = model_config['step_size']
    train_info['data']['duration'] = model_config['duration']
    train_info['data']['anomaly_sequence'] = model_config['anomaly_sequence']
    train_info['data']['data_entries'] = len(anomaly_label_array)

    train_info_json = json.dumps(train_info)
    print(train_info)
    try:
        response = requests.post(f'{train_info["settings"]["API_DATA_PROCESSING_URL"]}/preprocess_cgnn_train_data',
                                 files=train_files, data={'train_info': train_info_json})
        flask_response = Response(
            response.content,
            status=response.status_code,
            content_type=response.headers['Content-Type']
        )
        return flask_response
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Updates the configuration for the monitoring application.
    """
    new_config = request.json
    try:
        set_config(new_config)
        return jsonify({"success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_config', methods=['GET'])
def get_config_route():
    """
    Retrieves the current configuration.

    Returns:
        json: The current configuration.
    """
    config = get_config()
    return jsonify(config)


@app.route('/get_available_datasets', methods=['GET'])
def get_available_datasets():
    """
    Retrieves the available datasets.

    Returns:
        json: A dictionary of available datasets and their details.
    """
    combined_details = {}

    # Iterate over each directory in the datasets path
    for dir_name in os.listdir('datasets'):
        dir_path = os.path.join('datasets', dir_name)
        if os.path.isdir(dir_path):
            details_file = os.path.join(dir_path, 'details.json')
            if os.path.isfile(details_file):
                with open(details_file, 'r') as file:
                    details = json.load(file)
                    combined_details[dir_name] = details

    return jsonify(combined_details)


if __name__ == '__main__':
    set_initial_config()
    app.run(debug=True, host='0.0.0.0', port=5005)
