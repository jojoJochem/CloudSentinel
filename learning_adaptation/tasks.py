import numpy as np
import json
import logging
import os
import traceback
from celery import Celery

from cgnn.train import train
from cgnn.evaluate_prediction import predict_and_evaluate

logger = logging.getLogger(__name__)

celery = Celery('tasks', backend='redis://localhost:6379/2', broker='redis://localhost:6379/2')

# celery -A tasks worker --loglevel=info


@celery.task(bind=True)
def train_and_evaluate_task(self, train_array, test_array, anomaly_label_array, train_info):
    try:
        logger.info("Starting the training process.")
        model, model_config = train(train_info['data'], np.array(train_array, dtype=np.float32), np.array(test_array, dtype=np.float32),
                                    np.array(anomaly_label_array, dtype=np.float32))
        logger.info("Training completed.")

        logger.info("Starting the evaluation process.")
        predict_and_evaluate(model_config, model, np.array(train_array, dtype=np.float32),
                             np.array(test_array, dtype=np.float32), np.array(anomaly_label_array, dtype=np.float32))
        logger.info("Evaluation completed.")

        os.makedirs(f"trained_models_temp/{model_config['dataset']}_{model_config['id']}", exist_ok=True)
        with open(f"trained_models_temp/{model_config['dataset']}_{model_config['id']}/model_params.json", "w") as f:
            json.dump(train_info['data'], f, indent=2)

        # model_data = {
        #     'model': model_json,
        #     'model_params': train_info['data'],
        #     'model_config': model_config,
        #     'model_evaluation': model_evaluation
        # }
        return "Training Successful"
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)
