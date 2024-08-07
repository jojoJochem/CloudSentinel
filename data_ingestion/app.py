from flask import Flask, request, jsonify
import logging
import requests
import json
import traceback
import time
from celery import Celery
from kubernetes import client, config
from flask_cors import CORS

from data_collector import collect_crca_data, fetch_metrics
from config import set_initial_metric_config, get_config, set_config

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure and initialize Celery
app.config['broker_url'] = 'redis://redis:6379/0'
app.config['result_backend'] = 'redis://redis:6379/0'

# testing purposes
# app.config['broker_url'] = 'redis://localhost:6379/0'
# app.config['result_backend'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['broker_url'])
celery.conf.update(
    app.config,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_acks_late=True,  # If you are using late acknowledgments
    worker_prefetch_multiplier=1,  # Example configuration to avoid over-fetching
)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Command to run the Celery worker:
# celery -A app.celery worker --loglevel=info


@celery.task(bind=True)
def monitoring_task(self, monitor_info, iteration=0):
    """
    Celery task to perform continuous monitoring.

    Args:
        self (Task): The Celery task instance.
        monitor_info (dict): Information required for monitoring.
        iteration (int): The current iteration of the task.

    Returns:
        None
    """
    test_info = monitor_info
    test_info['task_id'] = self.request.id
    try:
        # Check for task revocation by querying the backend
        task = monitoring_task.AsyncResult(self.request.id)
        if task.state == 'REVOKED':
            logger.info(f"Task {self.request.id} revoked")
            return
        
        end_time = int(time.time())
        start_time = end_time - (test_info['data']['duration'] * 60)
        dataframe = fetch_metrics(test_info['data']['containers'], test_info['data']['metrics'], start_time, end_time,
                                    test_info['settings']['PROMETHEUS_URL'], test_info['data']['data_interval'])
        test_files = {'test_array': dataframe.to_csv(header=False, index=False)}
        test_info['data']['start_time'] = start_time
        test_info['data']['end_time'] = end_time
        test_info['data']['iteration'] = iteration
        test_info_json = json.dumps(test_info)
        logger.info(f"Sending data for task {self.request.id}, iteration {iteration}")
        requests.post(f"{test_info['settings']['API_DATA_PROCESSING_URL']}/preprocess_cgnn_data",
                      files=test_files, data={'test_info': test_info_json})

        # Schedule next iteration
        monitoring_task.apply_async(args=[monitor_info, iteration + 1], countdown=monitor_info['data']['test_interval'] * 60)

    except Exception:
        logger.error(f"Error in monitoring task {self.request.id}: {traceback.format_exc()}")


@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """
    Starts a new monitoring task.

    Returns:
        Response: JSON response with the status and task ID.
    """
    try:
        monitor_info_json = request.form.get('monitor_info')
        monitor_info = json.loads(monitor_info_json)
        task = monitoring_task.apply_async(args=[monitor_info])
        logger.info(f"Task {task.id} started")
        return jsonify({'status': 'monitoring_started', 'task_id': task.id}), 200
    except Exception as e:
        logger.error(f"Error starting monitoring task: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_active_tasks', methods=['GET'])
def get_tasks():
    """
    Retrieves the list of active tasks.

    Returns:
        Response: JSON response containing the active tasks.
    """
    try:
        active_tasks = celery.control.inspect().active()
        logger.info("Retrieved active tasks")
        return jsonify(active_tasks), 200
    except Exception as e:
        logger.error(f"Error retrieving active tasks: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/stop_monitoring/<task_id>', methods=['DELETE'])
def stop_monitoring(task_id):
    """
    Stops a running monitoring task.

    Args:
        task_id (str): The ID of the task to be stopped.

    Returns:
        Response: JSON response with the status of the operation.
    """
    try:
        if task_id:
            celery.control.revoke(task_id, terminate=True)
            logger.info(f"Task {task_id} stopped")
            return jsonify({'status': f'monitoring_stopped for task_id {task_id}'}), 200
        else:
            logger.warning("Task ID is missing")
            return jsonify({'status': 'task_id_missing'}), 400
    except Exception as e:
        logger.error(f"Error stopping task {task_id}: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/anomaly_rca', methods=['POST'])
def anomaly_rca():
    """
    Endpoint for anomaly root cause analysis (RCA).

    Returns:
        Response: JSON response with the RCA result.
    """
    try:
        data = request.form.get('crca_data')
        data = json.loads(data)
        response = collect_crca_data(data)
        logger.info("Anomaly RCA completed")
        return response
    except Exception as e:
        logger.error(f"Error performing anomaly RCA: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_pod_names', methods=['POST'])
def get_pod_names():
    """
    List the names of the pods in the specified namespace.

    Returns:
        Response: JSON response with the list of pod names.
    """
    try:
        namespace = request.json['namespace']
        v1 = client.CoreV1Api()
        ret = v1.list_namespaced_pod(namespace, watch=False)
        # ret = v1.list_pod_for_all_namespaces(watch=False)
        pod_names = [item.metadata.name for item in ret.items]
        logger.info(f"Retrieved pod names for namespace {namespace}")
        return jsonify(pod_names), 200
    except Exception as e:
        logger.error(f"Error retrieving pod names: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/load_kube_config', methods=['POST'])
def load_config():
    """
    Loads Kubernetes configuration.

    Returns:
        Response: JSON response with the status of the operation.
    """
    try:
        config.load_incluster_config()
        logger.info("Kubernetes configuration loaded successfully")
        return jsonify({"message": "Kubernetes configuration loaded successfully."}), 200
    except FileNotFoundError as e:
        logger.error(f"Kubernetes configuration file not found: {traceback.format_exc()}")
        return jsonify({"error": f"Kubernetes configuration file not found: {str(e)}"}), 400
    except config.ConfigException as e:
        logger.error(f"Error loading Kubernetes configuration: {traceback.format_exc()}")
        return jsonify({"error": f"Invalid Kubernetes configuration: {str(e)}"}), 400
    except ConnectionError as e:
        logger.error(f"Could not connect to Kubernetes cluster: {traceback.format_exc()}")
        return jsonify({"error": f"Could not connect to Kubernetes cluster: {str(e)}"}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route('/get_metrics', methods=['GET'])
def get_config_route():
    """
    Retrieves the current metrics configuration.

    Returns:
        Response: JSON response with the metrics configuration.
    """
    try:
        config = get_config()
        logger.info("Metrics configuration retrieved")
        return jsonify(config), 200
    except Exception as e:
        logger.error(f"Error retrieving metrics configuration: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Updates the configuration for the monitoring application.

    Returns:
        Response: JSON response with the status of the operation.
    """
    new_config = request.json
    try:
        set_config(new_config)
        logger.info("Configuration updated successfully")
        return jsonify("success"), 200
    except Exception as e:
        logger.error(f"Error updating configuration: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    set_initial_metric_config()
    logger.info("Starting Flask app")
    app.run(debug=False, host='0.0.0.0', port=5001)
