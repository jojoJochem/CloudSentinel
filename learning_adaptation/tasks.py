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
# celery = Celery(__name__, backend='redis://localhost:6379/2', broker='redis://localhost:6379/2')


class CustomTask(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 5, 'countdown': 60}
    time_limit = 7200
    soft_time_limit = 7000


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
        model, model_config = train(train_info['data'], np.array(train_array, dtype=np.float32),
                                    np.array(test_array, dtype=np.float32), np.array(anomaly_label_array, dtype=np.float32))
        logger.info("Training completed.")

        logger.info("Starting the evaluation process.")
        predict_and_evaluate(model_config, model, np.array(train_array, dtype=np.float32),
                             np.array(test_array, dtype=np.float32), np.array(anomaly_label_array, dtype=np.float32))
        logger.info("Evaluation completed.")

        # Create directory for saving model parameters
        model_dir = f"trained_models_temp/{model_config['dataset']}_{model_config['id']}"
        os.makedirs(model_dir, exist_ok=True)

        # Save model parameters to JSON file
        with open(f"{model_dir}/model_params.json", "w") as f:
            json.dump(train_info['data'], f, indent=2)

        return "Training Successful"
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)
