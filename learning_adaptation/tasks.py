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

        # Update task state to EVALUATING
        self.update_state(state='INITIATING', meta='Initiating training')

        def progress_callback(epoch, total_epochs, progress, total_steps):
            self.update_state(state='TRAINING', meta={
                'epoch': epoch + 1,
                'total_epochs': total_epochs,
                'progress': progress + 1,
                'total_steps': total_steps
            })

        # Train the model
        _, model_config = train(
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

        # Evaluate the model
        predict_and_evaluate(
            model_config,
            np.array(train_array, dtype=np.float32),
            np.array(test_array, dtype=np.float32),
            np.array(anomaly_label_array, dtype=np.float32)
        )

        logger.info("Evaluation completed.")

        # Save model parameters
        model_dir = f"trained_models_temp/{model_config['dataset']}_{model_config['id']}"
        os.makedirs(model_dir, exist_ok=True)
        with open(f"{model_dir}/model_params.json", "w") as f:
            json.dump(train_info['data'], f, indent=2)

        return "Training Successful"
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)
