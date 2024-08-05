# tasks.py
from celery import Celery
import pandas as pd
import logging
import traceback

from crca import run_crca

# Configure and initialize Celery
# celery = Celery(__name__, backend='redis://redis:6379/1', broker='redis://redis:6379/1')

# testing purposes
celery = Celery(__name__, backend='redis://localhost:6379/1', broker='redis://localhost:6379/1')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@celery.task(bind=True)
def run_crca_task(self, crca_data_json, crca_info):
    """
    Celery task to perform CRCA.

    Args:
        self (Task): The Celery task instance.
        crca_data_json (dict): Data to perform CRCA on.
        crca_info (dict): Information required for CRCA.

    Returns:
        dict: CRCA results
    """
    try:
        crca_data = pd.read_json(crca_data_json, orient='split')
        task_id = self.request.id

        response_data = run_crca(crca_data, crca_info, task_id)
        logger.info(response_data)
        return response_data
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)
