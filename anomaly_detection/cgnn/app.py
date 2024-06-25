from flask import Flask, request, jsonify
import logging
import torch
import os
import shutil
import json
import pandas as pd
import numpy as np
import traceback
import requests
from collections import OrderedDict
from flask_cors import CORS

from config import set_config, get_config, set_initial_config
from predict import load_model_and_predict

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure logging is set up
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """
    Endpoint to detect anomalies in the given data.

    Expects:
    - A CSV file containing the test data as 'test_array' in form-data.
    - JSON data 'test_info' in form-data containing:
      - 'data': {
          'model': <model_name>,
          'crca_threshold': <threshold_value>,
          'iteration': <iteration_number>,
          'start_time': <start_time>,
          'end_time': <end_time>,
          'crca_pods': <list_of_pods_for_crca>,
          'metrics': <list_of_metrics>,
          'data_interval': <interval_between_data_points>
        },
      - 'task_id': <task_id>

    Saves results to 'results/<task_id>/cgnn_results.json'.

    If anomaly_result > crca_threshold, triggers '/anomaly_rca' endpoint.

    Returns:
    - "success" if successful.
    - Error message and status code 500 if an error occurs.
    """
    try:
        test_data = pd.read_csv(request.files['test_array'], header=None).to_numpy(dtype=np.float32)
        test_info_json = request.form.get('test_info')
        test_info = json.loads(test_info_json)
        model = test_info['data']['model']
        anomaly_result = load_model_and_predict(test_data, model)
        crca_threshold = test_info['data']['crca_threshold']
        iteration = test_info['data']['iteration']
        task_id = test_info['task_id']

        path = 'results/' + task_id
        if not os.path.isdir(path):
            os.mkdir(path)

        if os.path.exists(path+'/cgnn_results.json'):
            with open(path+'/cgnn_results.json', 'r') as f:
                data = json.load(f)
            data['results'][iteration] = {
                'start_time': test_info['data']['start_time'],
                'end_time': test_info['data']['end_time'],
                'percentage': anomaly_result
            }
        else:
            data = {}
            data['model'] = model
            data['start_time'] = test_info['data']['start_time']
            data['containers'] = test_info['data']['containers']
            data['metrics'] = test_info['data']['metrics']
            data['step'] = test_info['data']['data_interval']
            data['crca_threshold'] = crca_threshold
            data['results'] = {
                iteration: {
                    'start_time': test_info['data']['start_time'],
                    'end_time': test_info['data']['end_time'],
                    'percentage': anomaly_result
                }
            }

        if anomaly_result > crca_threshold:
            crca_data = {
                'settings': test_info['settings'],
                'data': {
                    'task_id': task_id,
                    'start_time': test_info['data']['start_time'],
                    'end_time': test_info['data']['end_time'],
                    'containers': test_info['data']['crca_pods'],
                    'metrics': test_info['data']['metrics'],
                    'step': test_info['data']['data_interval']
                }
            }
            crca_data_json = json.dumps(crca_data)

            response = requests.post(f"{test_info['settings']['API_DATA_INGESTION_URL']}/anomaly_rca",
                                     data={'crca_data': crca_data_json})
            task_data = response.json()
            task_id = task_data.get('task_id')
            if task_id:
                data['results'][iteration]['crca_task_id'] = task_id

        with open(path+'/cgnn_results.json', 'w') as f:
            json.dump(data, f, indent=2)

        return "success"
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/save_model', methods=['POST'])
def save_model():
    """
    Endpoint to save a trained model and its related information.

    Expects:
    - The model file as 'model' in form-data.
    - JSON data 'model_info' in form-data containing:
      - 'data': {
          <model_name>: {
              'model_params': <model_parameters>,
              'model_config': <model_configuration>,
              'model_evaluation': <model_evaluation>
          }
        }

    Saves model to 'trained_models/<model_name>/model.pt' and related information
    to 'trained_models/<model_name>/model_params.json',
    'trained_models/<model_name>/model_config.json',
    'trained_models/<model_name>/model_evaluation.json'.

    Returns:
    - "success" if successful.
    - Error message and status code 500 if an error occurs.
    """
    try:
        trained_model = request.files['model']
        model_info_json = request.form.get('model_info')
        model_info = json.loads(model_info_json)
        model = next(iter(model_info['data']))

        model_path = 'trained_models/' + model

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        model_json = trained_model.read().decode('utf-8')
        model_dict = json.loads(model_json)
        loaded_model = OrderedDict({key: torch.tensor(value) for key, value in model_dict.items()})
        torch.save(loaded_model, model_path+'/model.pt')
        with open(model_path+'/model_params.json', 'w') as f:
            json.dump(model_info['data'][model]['model_params'], f, indent=2)
        with open(model_path+'/model_config.json', 'w') as f:
            json.dump(model_info['data'][model]['model_config'], f, indent=2)
        with open(model_path+'/model_evaluation.json', 'w') as f:
            json.dump(model_info['data'][model]['model_evaluation'], f, indent=2)

        return jsonify("success"), 200
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/get_config', methods=['GET'])
def get_config_route():
    """
    Endpoint to get the current configuration settings.

    Returns:
    - JSON object containing the current configuration.
    - Error message and status code 500 if an error occurs.
    """
    try:
        config = get_config()
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Endpoint to update the configuration settings.

    Expects:
    - JSON data containing the updated configuration.

    Returns:
    - "success" if successful.
    - Error message and status code 500 if an error occurs.
    """
    new_config = request.json
    try:
        set_config(new_config)
        return jsonify({"success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_available_models', methods=['GET'])
def get_available_models():
    """
    Endpoint to get information about available trained models.

    Returns:
    - JSON object containing information about each trained model.
    - Error message and status code 500 if an error occurs.
    """
    try:
        model_dir = 'trained_models/'
        models = {}
        for model in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model)
            if os.path.isdir(model_path):
                models[model] = {}
                with open(model_path+'/model_params.json', 'r') as f:
                    models[model]['model_params'] = json.load(f)
                with open(model_path+'/model_config.json', 'r') as f:
                    models[model]['model_config'] = json.load(f)
                with open(model_path+'/model_evaluation.json', 'r') as f:
                    models[model]['model_evaluation'] = json.load(f)
        return jsonify(models), 200
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/get_results', methods=['POST'])
def get_results():
    """
    Endpoint to get results for a specific task ID.

    Expects:
    - JSON data containing 'task_id' for which results are requested.

    Returns:
    - JSON object containing results for the specified task ID.
    - Error message and status code 500 if an error occurs.
    """
    try:
        task_id = request.json['task_id']
        with open('results/'+task_id+'/cgnn_results.json', 'r') as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_all_results', methods=['GET'])
def get_all_results():
    """
    Endpoint to get results for all tasks.

    Returns:
    - JSON object containing results for all tasks.
    - Error message and status code 500 if an error occurs.
    """
    try:
        task_ids = os.listdir('results/')
        results = {}
        for task_id in task_ids:
            result_file = 'results/'+task_id+'/cgnn_results.json'
            if os.path.isfile(result_file):
                with open('results/'+task_id+'/cgnn_results.json', 'r') as f:
                    data = json.load(f)
                results[task_id] = data
        return jsonify(results), 200
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/delete_results', methods=['POST'])
def delete_results():
    """
    Endpoint to delete results for a specific task ID.

    Expects:
    - JSON data containing 'taskId' and 'crcaLink' for which results are to be deleted.

    Returns:
    - JSON object indicating success.
    - Error message and status code 500 if an error occurs.
    """
    try:
        data = request.get_json()
        task_id = data['taskId']
        crca_link = data['crcaLink']
        with open('results/'+task_id+'/cgnn_results.json', 'r') as f:
            data = json.load(f)
        for key, result in data['results'].items():
            if 'crca_task_id' in result:
                requests.delete(f"{crca_link}/delete_results/{result['crca_task_id']}")
        shutil.rmtree('results/'+task_id)
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    set_initial_config()
    app.run(debug=True, host='0.0.0.0', port=5013)
