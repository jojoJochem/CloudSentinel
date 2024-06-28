from flask import Flask, request, jsonify
import logging
import json
import os
import traceback
import shutil
import pandas as pd
from flask_cors import CORS
from celery import Celery

from config import set_config, get_config, set_initial_config
from crca import run_crca

# Set multiprocessing start method to 'spawn' for Celery compatibility
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Set the environment variable
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Configure and initialize Celery instance
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/1'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/1'
celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'], broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Command to run the Celery worker:
# celery -A app.celery worker --loglevel=info -P gevent


@celery.task(bind=True)
def run_crca_task(self, crca_data_json, crca_info):
    """
    Celery task to perform crca.

    Args:
        self (Task): The Celery task instance.
        crca_data_json (dict): Data to perform crca on.
        crca_info (dict): Information required for crca.

    Returns:
        None
    """
    try:
        crca_data = pd.read_json(crca_data_json, orient='split')
        task_id = self.request.id

        run_crca(crca_data, crca_info, task_id)
        return "success"
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)


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
        logger.info("Received request to initiate CRCA task")
        crca_file = request.files['crca_file']
        crca_data = pd.read_csv(crca_file, header=None)
        crca_data_json = crca_data.to_json(orient='split')
        crca_info_json = request.form.get('crca_info')
        crca_info = json.loads(crca_info_json)

        # Initiate CRCA task asynchronously
        task = run_crca_task.apply_async(args=[crca_data_json, crca_info])
        logger.info(f"CRCA task initiated with task_id: {task.id}")

        return jsonify({"task_id": task.id}), 202

    except Exception as e:
        logger.error(f"Error initiating CRCA task: {traceback.format_exc()}")
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
        logger.info(f"Received request to get status for task_id: {task_id}")
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
        logger.info(f"Status for task_id {task_id}: {response}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error retrieving status for task_id {task_id}: {traceback.format_exc()}")
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
        logger.info(f"Received request to get results for task_id: {task_id}")
        with open('results/'+task_id+'/crca_results.json', 'r') as f:
            data = json.load(f)
        logger.info(f"Results retrieved for task_id: {task_id}")
        return jsonify(data), 200

    except Exception as e:
        logger.error(f"Error retrieving results for task_id {task_id}: {traceback.format_exc()}")
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
        logger.info(f"Received request to delete results for task_id: {task_id}")
        shutil.rmtree('results/'+task_id)  # Delete directory recursively
        logger.info(f"Results deleted for task_id: {task_id}")
        return jsonify({"success": True}), 200

    except Exception as e:
        logger.error(f"Error deleting results for task_id {task_id}: {traceback.format_exc()}")
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
        logger.info("Received request to get configuration")
        config = get_config()
        logger.info("Configuration retrieved successfully")
        return jsonify(config), 200

    except Exception as e:
        logger.error(f"Error retrieving configuration: {traceback.format_exc()}")
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
        logger.info("Received request to update configuration")
        set_config(new_config)
        logger.info("Configuration updated successfully")
        return jsonify({"status": "success", "message": "Configuration updated successfully"}), 200

    except Exception as e:
        logger.error(f"Error updating configuration: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    logger.info("Initializing configuration")
    set_initial_config()
    logger.info("Starting Flask app")
    app.run(debug=True, host='0.0.0.0', port=5023)
