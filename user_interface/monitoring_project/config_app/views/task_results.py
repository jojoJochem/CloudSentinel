import json
import requests
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from requests.exceptions import RequestException
from io import StringIO
from django.http import JsonResponse
import logging
import traceback

from .utils import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_results(request, task_id, task_type):
    """
    Fetches results based on task type and renders the appropriate page.

    Args:
        request (HttpRequest): The request object.
        task_id (str): The ID of the task to fetch results for.
        task_type (str): The type of task ('crca' or 'training').

    Returns:
        HttpResponse: The rendered results page.
    """
    logger.info(f"Fetching results for task_id: {task_id}, task_type: {task_type}")
    match task_type:
        case 'crca':
            return crca_result(request, task_id)
        case 'training':
            return training_result(request)


def crca_result(request, task_id):
    """
    Fetches and displays the results for a CRCA task.

    Args:
        request (HttpRequest): The request object.
        task_id (str): The ID of the CRCA task.

    Returns:
        HttpResponse: The rendered CRCA results page.
        HttpResponseRedirect: Redirects to the home page on failure.
    """
    try:
        logger.info(f"Fetching CRCA results for task_id: {task_id}")
        response = requests.get(f'{settings.API_CRCA_ANOMALY_DETECTION_URL}/results/{task_id}')
        response.raise_for_status()
        response_data = response.json()
        graph_images = response_data.get('graph_image', [])
        csv_data = response_data.get('ranking', '')
        csv_df = pd.read_csv(StringIO(csv_data))
        logger.info("CRCA results retrieved successfully")

        return render(request, 'config_app/anomaly_detection/crca/crca_display_response.html', {
            'message': 'Processing complete!',
            'graph_images': graph_images,
            'ranking': csv_df.to_html(classes='table table-striped', index=False)
        })

    except RequestException as e:
        logger.error(f"Failed to retrieve CRCA results: {traceback.format_exc()}")
        messages.error(request, f'Failed to retrieve results: {str(e)}')
        return redirect('home')


def training_result(request):
    """
    Handles the display and submission of training results.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered training results page or home page on successful submission.
        JsonResponse: A JSON response indicating success or failure.
    """
    if request.method == 'POST':
        selected_model = request.POST.get('selected_model')

        if selected_model:
            try:
                logger.info("Fetching available models for training")
                response = requests.get(f'{settings.API_LEARNING_ADAPTATION_URL}/get_available_models')
                response.raise_for_status()
                models = response.json()

                if selected_model in models:
                    model_info = models[selected_model]
                    model_info_json = json.dumps({
                        'settings': get_settings(),
                        'data': {selected_model: model_info}
                    })

                    try:
                        logger.info(f"Saving model {selected_model} to detection module")
                        response = requests.post(
                            f'{settings.API_LEARNING_ADAPTATION_URL}/save_to_detection_module',
                            data={'model_info': model_info_json}
                        )
                        response.raise_for_status()
                        logger.info(f"Model {selected_model} saved successfully")
                        return redirect('home')
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Failed to save model to detection module: {traceback.format_exc()}")
                        return JsonResponse({'status': 'error', 'message': str(e)})
            except RequestException as e:
                logger.error(f"Failed to fetch available models: {traceback.format_exc()}")
                return JsonResponse({'status': 'error', 'message': str(e)})
        else:
            logger.warning("No model selected")
            return JsonResponse({'status': 'error', 'message': 'No model selected'})

    try:
        logger.info("Fetching available models for rendering training results page")
        response = requests.get(f'{settings.API_LEARNING_ADAPTATION_URL}/get_available_models')
        response.raise_for_status()
        models = response.json()
        context = {
            'models': models
        }
        logger.info("Rendering training results page")
        return render(request, 'config_app/training/cgnn_training_response.html', context)
    except RequestException as e:
        logger.error(f"Failed to fetch available models: {traceback.format_exc()}")
        return JsonResponse({'status': 'error', 'message': str(e)})
