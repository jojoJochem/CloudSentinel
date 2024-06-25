import numpy as np
import json
import logging
import os
import traceback
from celery import Celery

from cgnn.train import train
from cgnn.evaluate_prediction import predict_and_evaluate

# Initialize logger
logger = logging.getLogger(__name__)

# Configure Celery
celery = Celery('tasks', backend='redis://redis:6379/2', broker='redis://redis:6379/2')

# Command to run the Celery worker:
# celery -A tasks worker --loglevel=info


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
