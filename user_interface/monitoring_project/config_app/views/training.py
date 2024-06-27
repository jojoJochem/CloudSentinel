from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from requests.exceptions import RequestException
import json
import requests
import logging
import traceback

from .utils import get_config, get_settings, get_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_algorithms(request):
    """
    Renders the home page for training algorithms.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered training algorithms home page.
    """
    logger.info("Rendering training algorithms home page")
    return render(request, 'config_app/training/train_algorithms_home.html')


def train_cgnn(request):
    """
    Renders the CGNN training home page with configuration data.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered CGNN training home page with config data.
    """
    logger.info("Fetching CGNN training configuration")
    config_data = get_config(settings.API_LEARNING_ADAPTATION_URL)
    return render(request, 'config_app/training/cgnn_train_home.html', {'config': config_data})


def cgnn_train_data(request):
    """
    Handles CGNN training data submission and renders the waiting page.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered page to submit CGNN training data.
        HttpResponseRedirect: Redirects to the CGNN training data page on failure.
    """
    if request.method == 'POST':
        selected_dataset = request.POST.get('dataset')
        selected_containers = request.POST.getlist('containers')
        selected_metrics = request.POST.getlist('metrics')
        comment = request.POST.get('comment')

        cgnn_info = {
            'dataset': selected_dataset,
            'comment': comment,
            'containers': selected_containers,
            'metrics': selected_metrics
        }
        train_info = {
            'settings': get_settings(),
            'data': cgnn_info
        }
        train_info_json = json.dumps(train_info)
        try:
            logger.info("Submitting CGNN training data")
            response = requests.post(settings.API_LEARNING_ADAPTATION_URL + '/cgnn_train_with_existing_dataset',
                                     data={'train_info': train_info_json})
            response.raise_for_status()
            response_data = response.json()
            task_id = response_data.get('task_id')
            logger.info(f"CGNN training task {task_id} started")
            return render(request, 'config_app/waiting_page.html', {'task_id': task_id,
                                                                    'api_url': settings.API_LEARNING_ADAPTATION_URL,
                                                                    'task_type': 'training'})
        except RequestException as e:
            logger.error(f"Failed to train CGNN: {traceback.format_exc()}")
            messages.error(request, f'Failed to train CGNN: {str(e)}')
            return redirect('cgnn_train_data')

    datasets = []

    try:
        logger.info("Fetching available datasets for CGNN training")
        response = requests.get(settings.API_LEARNING_ADAPTATION_URL + '/get_available_datasets')
        response.raise_for_status()
        details = response.json()
        for key, value in details.items():
            datasets.append({
                'name': key,
                'data': value
            })
        datasets_json = json.dumps(datasets)
        logger.info("Available datasets retrieved successfully")
    except RequestException as e:
        logger.error(f"Failed to get available datasets: {traceback.format_exc()}")
        messages.error(request, f'Failed to get available datasets: {str(e)}')
        datasets_json = json.dumps(datasets)

    return render(request, 'config_app/training/cgnn_train_data.html', {
        'datasets': datasets,
        'datasets_json': datasets_json
    })


def upload_cgnn_train_data(request):
    """
    Handles the upload of CGNN training data and initiates the training process.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered waiting page with task details.
        HttpResponseRedirect: Redirects to the CGNN training page on failure.
    """
    if request.method == 'POST':
        train_array = request.FILES['train_array']
        test_array = request.FILES['test_array']
        anomaly_label_array = request.FILES['anomaly_label_array']
        anomaly_sequence = request.POST.get('anomaly_sequence') == 'on'
        dataset = request.POST.get('dataset')
        comment = request.POST.get('comment')
        metrics = json.loads(request.POST.get('selected_metrics', '[]'))
        step_size = int(request.POST.get('step_size', 1))
        duration = int(request.POST.get('duration', 1))
        containers = json.loads(request.POST.get('containers', '[]'))

        train_files = {
            'train_array': train_array,
            'test_array': test_array,
            'anomaly_label_array': anomaly_label_array
        }
        cgnn_info = {
            'anomaly_sequence': anomaly_sequence,
            'dataset': dataset,
            'comment': comment,
            'containers': containers,
            'metrics': metrics,
            'step_size': step_size,
            'duration': duration
        }
        train_info = {
            'settings': get_settings(),
            'data': cgnn_info
        }
        train_info_json = json.dumps(train_info)
        try:
            logger.info("Uploading CGNN training data")
            response = requests.post(f'{settings.API_DATA_PROCESSING_URL}/preprocess_cgnn_train_data',
                                     files=train_files, data={'train_info': train_info_json})
            response.raise_for_status()
            response_data = response.json()
            task_id = response_data.get('task_id')
            logger.info(f"CGNN training task {task_id} started")
            return render(request, 'config_app/waiting_page.html', {'task_id': task_id,
                                                                    'api_url': settings.API_LEARNING_ADAPTATION_URL,
                                                                    'task_type': 'training'})
        except RequestException as e:
            logger.error(f"Failed to train CGNN: {traceback.format_exc()}")
            messages.error(request, f'Failed to train CGNN: {str(e)}')
            return redirect('train_cgnn')

    metrics = get_metrics()
    logger.info("Rendering CGNN training data upload page")
    return render(request, 'config_app/training/cgnn_upload_train_data.html', {'metrics': metrics})
