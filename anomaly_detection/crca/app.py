# app.py
from flask import Flask, request, jsonify
import logging
import json
import traceback
import shutil
from celery import Celery
import pandas as pd
import multiprocessing
from flask_cors import CORS

from config import set_config, get_config, set_initial_config
from tasks import run_crca_task

multiprocessing.set_start_method('spawn', force=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/1',
    CELERY_RESULT_BACKEND='redis://localhost:6379/1'
)

celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'], broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/crca', methods=['POST'])
def crca():
    try:
        logging.info("Received request: %s", request)
        crca_file = request.files['crca_file']
        crca_data = pd.read_csv(crca_file, header=None)
        crca_data_json = crca_data.to_json(orient='split')  # Convert DataFrame to JSON string
        crca_info_json = request.form.get('crca_info')
        crca_info = json.loads(crca_info_json)
        task = run_crca_task.apply_async(args=[crca_data_json, crca_info])
        return jsonify({"task_id": task.id}), 202
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/get_status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = run_crca_task.AsyncResult(task_id)
    print(task.state)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return jsonify(response)


@app.route('/results/<task_id>', methods=['GET'])
def get_results(task_id):
    try:
        with open('results/'+task_id+'/crca_results.json', 'r') as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete_results/<task_id>', methods=['DELETE'])
def delete_results(task_id):
    try:
        shutil.rmtree('results/'+task_id)
        return jsonify({"success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_config', methods=['GET'])
def get_config_route():
    try:
        config = get_config()
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Updates the configuration for the monitoring application.
    """
    new_config = request.json
    try:
        set_config(new_config)
        return jsonify({"status": "success", "message": "Configuration updated successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    set_initial_config()
    app.run(debug=True, host='0.0.0.0', port=5023)
