import json
import requests
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from requests.exceptions import RequestException
from io import StringIO
from django.http import JsonResponse

from .utils import get_settings


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
        response = requests.get(f'{settings.API_CRCA_ANOMALY_DETECTION_URL}/results/{task_id}')
        response.raise_for_status()
        response_data = response.json()
        graph_images = response_data.get('graph_image', [])
        csv_data = response_data.get('ranking', '')
        csv_df = pd.read_csv(StringIO(csv_data))

        return render(request, 'config_app/anomaly_detection/crca/crca_display_response.html', {
            'message': 'Processing complete!',
            'graph_images': graph_images,
            'ranking': csv_df.to_html(classes='table table-striped', index=False)
        })

    except RequestException as e:
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
                        response = requests.post(
                            f'{settings.API_LEARNING_ADAPTATION_URL}/save_to_detection_module',
                            data={'model_info': model_info_json}
                        )
                        response.raise_for_status()
                        return redirect('home')
                    except requests.exceptions.RequestException as e:
                        return JsonResponse({'status': 'error', 'message': str(e)})
            except RequestException as e:
                return JsonResponse({'status': 'error', 'message': str(e)})
        else:
            return JsonResponse({'status': 'error', 'message': 'No model selected'})

    try:
        response = requests.get(f'{settings.API_LEARNING_ADAPTATION_URL}/get_available_models')
        response.raise_for_status()
        models = response.json()
        context = {
            'models': models
        }
        return render(request, 'config_app/training/cgnn_training_response.html', context)
    except RequestException as e:
        return JsonResponse({'status': 'error', 'message': str(e)})
