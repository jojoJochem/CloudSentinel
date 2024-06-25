from flask import Flask, request, jsonify
from celery import Celery
import logging
import requests
import time
import json
from kubernetes import client, config
import traceback
from flask_cors import CORS

from data_collector import collect_crca_data, fetch_metrics
from config import set_initial_metric_config, get_config, set_config

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Celery
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Command to run the Celery worker:
# celery -A app_1.celery worker --loglevel=info


@celery.task(bind=True)
def monitoring_task(self, monitor_info):
    """
    Celery task to perform continuous monitoring.

    Args:
        self (Task): The Celery task instance.
        monitor_info (dict): Information required for monitoring.

    Returns:
        None
    """
    iteration = 0
    test_info = monitor_info
    test_info['task_id'] = self.request.id

    while True:
        # Check for task revocation by querying the backend
        task = monitoring_task.AsyncResult(self.request.id)
        if task.state == 'REVOKED':
            break

        end_time = int(time.time())
        start_time = end_time - (test_info['data']['duration'] * 60)
        dataframe = fetch_metrics(test_info['data']['pods'], test_info['data']['metrics'], start_time, end_time,
                                  test_info['settings']['PROMETHEUS_URL'], test_info['data']['data_interval'])
        test_files = {'test_array': dataframe.to_csv(header=False, index=False)}
        test_info['data']['start_time'] = start_time
        test_info['data']['end_time'] = end_time
        test_info['data']['iteration'] = iteration
        test_info_json = json.dumps(test_info)
        print(test_info_json)
        requests.post(f"{test_info['settings']['API_DATA_PROCESSING_URL']}/preprocess_cgnn_data",
                      files=test_files, data={'test_info': test_info_json})
        time.sleep(test_info['data']['test_interval'] * 60)
        iteration += 1


@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """
    Starts a new monitoring task.

    Returns:
        Response: JSON response with the status and task ID.
    """
    monitor_info_json = request.form.get('monitor_info')
    monitor_info = json.loads(monitor_info_json)
    task = monitoring_task.apply_async(args=[monitor_info])
    logging.info(f"Task {task.id} started")
    return jsonify({'status': 'monitoring_started', 'task_id': task.id}), 200


@app.route('/get_active_tasks', methods=['GET'])
def get_tasks():
    """
    Retrieves the list of active tasks.

    Returns:
        Response: JSON response containing the active tasks.
    """
    active_tasks = celery.control.inspect().active()
    return jsonify(active_tasks), 200


@app.route('/stop_monitoring/<task_id>', methods=['DELETE'])
def stop_monitoring(task_id):
    """
    Stops a running monitoring task.

    Args:
        task_id (str): The ID of the task to be stopped.

    Returns:
        Response: JSON response with the status of the operation.
    """
    if task_id:
        celery.control.revoke(task_id, terminate=True)
        return jsonify({'status': f'monitoring_stopped for task_id {task_id}'}), 200
    else:
        return jsonify({'status': 'task_id_missing'}), 400


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
        return response
    except Exception as e:
        logger.error(traceback.format_exc())
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
        pod_names = [item.metadata.name for item in ret.items]
        return jsonify(pod_names), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/load_kube_config', methods=['POST'])
def load_config():
    """
    Loads Kubernetes configuration.

    Returns:
        Response: JSON response with the status of the operation.
    """
    try:
        # config.load_kube_config(request.json['kube_config_path'])
        config.load_incluster_config()
        return jsonify({"message": "Kubernetes configuration loaded successfully."}), 200
    except FileNotFoundError as e:
        logging.error("Kubernetes configuration file not found: %s", str(e))
        return jsonify({"error": "Kubernetes configuration file not found"}), 400
    except config.ConfigException as e:
        logging.error("Error loading Kubernetes configuration: %s", str(e))
        return jsonify({"error": "Invalid Kubernetes configuration"}), 400
    except ConnectionError as e:
        logging.error("Could not connect to Kubernetes cluster: %s", str(e))
        return jsonify({"error": "Could not connect to Kubernetes cluster"}), 503
    except Exception as e:
        logging.error("Unexpected error: %s", str(e))
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/get_metrics', methods=['GET'])
def get_config_route():
    """
    Retrieves the current metrics configuration.

    Returns:
        Response: JSON response with the metrics configuration.
    """
    try:
        config = get_config()
        return jsonify(config), 200
    except Exception as e:
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
        return jsonify("success"), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    set_initial_metric_config()
    app.run(debug=True, host='0.0.0.0', port=5001)
