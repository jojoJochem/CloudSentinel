from flask import Flask, request, jsonify
import logging
import json
import traceback
import shutil
import pandas as pd
from flask_cors import CORS
from celery import Celery

from config import set_config, get_config, set_initial_config
from tasks import run_crca_task

# Set multiprocessing start method to 'spawn' for Celery compatibility
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Celery
app.config.update(
    CELERY_BROKER_URL='redis://redis:6379/1',      # Redis URL for broker
    CELERY_RESULT_BACKEND='redis://redis:6379/1'   # Redis URL for backend
)

# Initialize Celery instance
celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'], broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/crca', methods=['POST'])
def crca():
    """
    Endpoint to initiate CRCA (Change Root Cause Analysis) task.

    Expects:
    - 'crca_file': CSV file containing CRCA data.
    - 'crca_info': JSON string containing additional CRCA information.

    Returns:
    - JSON response with 'task_id' upon successful initiation.
    - Error response if an exception occurs.
    """
    try:
        logging.info("Received request: %s", request)
        crca_file = request.files['crca_file']
        crca_data = pd.read_csv(crca_file, header=None)
        crca_data_json = crca_data.to_json(orient='split')
        crca_info_json = request.form.get('crca_info')
        crca_info = json.loads(crca_info_json)

        # Initiate CRCA task asynchronously
        task = run_crca_task.apply_async(args=[crca_data_json, crca_info])

        return jsonify({"task_id": task.id}), 202

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/get_status/<task_id>', methods=['GET'])
def get_status(task_id):
    """
    Endpoint to retrieve the status of a specific task identified by task_id.

    Expects:
    - task_id: Unique identifier of the task.

    Returns:
    - JSON response with task state and result if available.
    - Error response if the task_id does not exist or an exception occurs.
    """
    try:
        task = run_crca_task.AsyncResult(task_id)

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

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/results/<task_id>', methods=['GET'])
def get_results(task_id):
    """
    Endpoint to retrieve results of a specific CRCA task identified by task_id.

    Expects:
    - task_id: Unique identifier of the task whose results are to be retrieved.

    Returns:
    - JSON response with CRCA results.
    - Error response if the task_id does not exist or an exception occurs.
    """
    try:
        with open('results/'+task_id+'/crca_results.json', 'r') as f:
            data = json.load(f)
        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete_results/<task_id>', methods=['DELETE'])
def delete_results(task_id):
    """
    Endpoint to delete results associated with a specific CRCA task identified by task_id.

    Expects:
    - task_id: Unique identifier of the task whose results are to be deleted.

    Returns:
    - JSON response indicating success upon successful deletion.
    - Error response if the task_id does not exist or an exception occurs.
    """
    try:
        shutil.rmtree('results/'+task_id)  # Delete directory recursively
        return jsonify({"success": True}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_config', methods=['GET'])
def get_config_route():
    """
    Endpoint to retrieve the current configuration settings.

    Returns:
    - JSON response with current configuration settings.
    - Error response if an exception occurs while retrieving the configuration.
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
    - JSON payload containing new configuration settings.

    Returns:
    - JSON response indicating success upon successful update.
    - Error response if an exception occurs while updating the configuration.
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
