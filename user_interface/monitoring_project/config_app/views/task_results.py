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
    match task_type:
        case 'crca':
            return crca_result(request, task_id)
        case 'training':
            return training_result(request)


def crca_result(request, task_id):
    try:
        response = requests.get(f'{settings.API_CRCA_ANOMALY_DETECTION_URL}/results/{task_id}')
        response_data = response.json()
        graph_images = response_data.get('graph_image', [])
        csv_data = response_data.get('ranking', '')
        csv_df = pd.read_csv(StringIO(csv_data))

        print(csv_data)

        return render(request, 'config_app/anomaly_detection/crca/crca_display_response.html', {
            'message': 'Processing complete!',
            'graph_images': graph_images,
            'ranking': csv_df.to_html(classes='table table-striped', index=False)
        })

    except RequestException as e:
        messages.error(request, f'Failed to retrieve results: {str(e)}')
        return redirect('home')


def training_result(request):
    if request.method == 'POST':
        selected_model = request.POST.get('selected_model')

        if selected_model:
            response = requests.get(f'{settings.API_LEARNING_ADAPTATION_URL}/get_available_models')
            response.raise_for_status()
            models = response.json()

            if selected_model in models:
                model_info = models[selected_model]
                model_info_json = json.dumps({
                    'settings': get_settings(),
                    'data': {selected_model: model_info}
                })

                print(model_info_json)

                try:
                    response = requests.post(
                        f'{settings.API_LEARNING_ADAPTATION_URL}/save_to_detection_module',
                        data={'model_info': model_info_json}
                    )
                    response.raise_for_status()
                    return redirect('home')
                except requests.exceptions.RequestException as e:
                    return JsonResponse({'status': 'error', 'message': str(e)})
        else:
            return JsonResponse({'status': 'error'})

    try:
        response = requests.get(f'{settings.API_LEARNING_ADAPTATION_URL}/get_available_models')
        response.raise_for_status()
        models = response.json()
        print(models)
        context = {
            'models': models
        }
        return render(request, 'config_app/training/cgnn_training_response.html', context)
    except RequestException as e:
        return JsonResponse({'status': 'error', 'message': str(e)})
