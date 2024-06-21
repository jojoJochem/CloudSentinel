import os
from crca import run_crca
import logging
from celery import Celery
import pandas as pd

# Set the environment variable
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

logger = logging.getLogger(__name__)

celery = Celery('tasks', backend='redis://localhost:6379/1', broker='redis://localhost:6379/1')

# celery -A tasks worker --loglevel=info -P gevent


@celery.task(bind=True)
def run_crca_task(self, crca_data_json, crca_info):
    try:
        crca_data = pd.read_json(crca_data_json, orient='split')
        task_id = self.request.id

        run_crca(crca_data, crca_info, task_id)
        return "success"
    except Exception as e:
        logger.error(e)
        raise self.retry(exc=e, countdown=60)
