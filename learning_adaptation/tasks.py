# tasks.py

import os
import json
import logging
import traceback
import numpy as np
from celery import Celery, Task
from cgnn.train import train
from cgnn.evaluate_prediction import predict_and_evaluate

# Configure and initialize Celery
celery = Celery(__name__, backend='redis://redis:6379/2', broker='redis://redis:6379/2')

# testing purposes
# celery = Celery(__name__, backend='redis://localhost:6379/2', broker='redis://localhost:6379/2')


class CustomTask(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 1, 'countdown': 60}
    time_limit = 10800
    soft_time_limit = 10000


celery.Task = CustomTask

celery.conf.update(
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_heartbeat=0,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@celery.task(bind=True)
def train_and_evaluate_task(self, train_array, test_array, anomaly_label_array, train_info):
    """
    Celery task to train and evaluate a CGNN model.

    Args:
        self (Task): The Celery task instance.
        train_array (list): Training data array.
        test_array (list): Testing data array.
        anomaly_label_array (list): Anomaly label array.
        train_info (dict): Training information and settings.

    Returns:
        str: Success message if training is completed successfully.

    Raises:
        Exception: Retries the task if an exception occurs.
    """
    try:
        logger.info("Starting the training process.")

        # Update task state to EVALUATING
        self.update_state(state='INITIATING', meta='Initiating training')

        def progress_callback(progress_state, message, outer=None, total_outer=None, inner=None, total_inner=None):
            if message == '':
                self.update_state(state=progress_state, meta={
                    'outer': outer + 1,
                    'total_outer': total_outer,
                    'inner': inner + 1,
                    'total_inner': total_inner
                })
            else:
                self.update_state(state=progress_state, meta=message)

        # Train the model
        model_config, feature_importance = train(
            train_info['data'],
            np.array(train_array, dtype=np.float32),
            np.array(test_array, dtype=np.float32),
            np.array(anomaly_label_array, dtype=np.float32),
            progress_callback=progress_callback
        )

        logger.info("Training completed.")

        # Update task state to EVALUATING
        self.update_state(state='EVALUATING', meta='Evaluating the model')

        logger.info("Starting the evaluation process.")
        # testing purposes
        # # Load model_config from trained_models_temp/SMD_1-1_30062024_095123 directory
        # model_dir = "trained_models_temp/SMD_1-1_01072024_152236"
        # with open(f"{model_dir}/model_config.json", "r") as f:
        #     model_config = json.load(f)

        # Evaluate the model
        predict_and_evaluate(
            model_config,
            np.array(train_array, dtype=np.float32),
            np.array(test_array, dtype=np.float32),
            np.array(anomaly_label_array, dtype=np.float32),
            progress_callback=progress_callback
        )

        logger.info("Evaluation completed.")
        if model_config['feature_importance']:
            # Generate feature names based on the given container and metric names
            feature_names = {}
            index = 0
            for container in train_info['data']["containers"]:
                for metric in train_info['data']["metrics"]:
                    feature_names[str(index)] = f"{container}_{metric}"
                    index += 1
            print(feature_importance)
            # Create a ranked list of features based on importance
            ranked_features = {name: feature_importance[int(idx)] for idx, name in feature_names.items()}
            ranked_features = dict(sorted(ranked_features.items(), key=lambda item: item[1], reverse=True))

            train_info['data']["ranked_features"] = ranked_features
        print(train_info['data'])

        # Save model parameters
        model_dir = f"trained_models_temp/{model_config['dataset']}_{model_config['id']}"
        os.makedirs(model_dir, exist_ok=True)
        with open(f"{model_dir}/model_params.json", "w") as f:
            json.dump(train_info['data'], f, indent=2)

        return "Training Successful"
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)
