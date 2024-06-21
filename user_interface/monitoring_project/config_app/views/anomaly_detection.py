import requests
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from requests.exceptions import RequestException
from io import StringIO
import json

from .utils import get_config, get_pods, get_available_models, get_settings, get_metrics


def home_anomaly_detection(request):
    return render(request, 'config_app/anomaly_detection/home_anomaly_detection.html')


def perform_anomaly_detection_cgnn(request):
    config_data = get_config(settings.API_CGNN_ANOMALY_DETECTION_URL)
    return render(request, 'config_app/anomaly_detection/cgnn/cgnn_anomaly_detection_home.html', {'config': config_data})


def perform_anomaly_detection_crca(request):
    config_data = get_config(settings.API_CRCA_ANOMALY_DETECTION_URL)
    return render(request, 'config_app/anomaly_detection/crca/crca_anomaly_detection_home.html', {'config': config_data})


def select_crca_file(request):
    pod_names_url = get_pods(settings.CLUSTER_NAMESPACE)
    try:
        response = requests.get(pod_names_url)
        response.raise_for_status()
        pod_names_data = response.json()
    except RequestException as e:
        pod_names_data = []
        messages.error(request, f'Failed to retrieve pod names: {str(e)}')
    return render(request, 'config_app/anomaly_detection/crca/crca_select_file.html', {'pod_names': pod_names_data})


def chosen_crca(request):
    if request.method == 'POST':
        selected_pod_names = request.POST.getlist('selected_pod_names')
        start_datetime = request.POST.get('start_datetime')
        end_datetime = request.POST.get('end_datetime')
        data = {
            'pod_names': selected_pod_names,
            'start_datetime': start_datetime,
            'end_datetime': end_datetime
        }
        try:
            response = requests.post(f'{settings.API_DATA_PROCESSING_URL}/crca_uploaded_by_user', json=data)
            response.raise_for_status()
            response_data = response.json()
            graph_images = response_data['graph_image']
            csv_data = response_data['file']
            csv_df = pd.read_csv(StringIO(csv_data))
            context = {
                'message': response_data['message'],
                'graph_images': graph_images,
                'csv_data': csv_df.to_html(classes='table table-striped', index=False)
            }
            return render(request, 'config_app/anomaly_detection/crca/display_crca_response.html', context)
        except requests.RequestException as e:
            messages.error(request, f'Failed to process data: {str(e)}')
            return redirect('select_crca_file')
    return redirect('select_crca_file')


def upload_cgnn_data(request):
    models = {}
    selected_model = None
    result = None

    models = get_available_models(settings.API_CGNN_ANOMALY_DETECTION_URL)

    if request.method == 'POST':
        selected_model = request.POST.get('selected_model')
        data_file = request.FILES.get('data_file')
        test_info = {
            'settings': get_settings(),
            'data': {'model': selected_model}
        }
        test_info_json = json.dumps(test_info)
        if selected_model and data_file:
            try:
                response = requests.post(f'{settings.API_DATA_PROCESSING_URL}/preprocess_cgnn_data',
                                         files={'test_array': data_file}, data={'test_info': test_info_json})
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as e:
                messages.error(request, f'Failed to perform anomaly detection: {str(e)}')

    context = {
        'models': models,
        'selected_model': selected_model,
        'result': result,
    }
    return render(request, 'config_app/anomaly_detection/cgnn/cgnn_upload_data.html', context)


def upload_crca_data(request):
    if request.method == 'POST':
        crca_file = request.FILES['crca_file']
        metrics = json.loads(request.POST.get('selected_metrics', '[]'))
        containers = json.loads(request.POST.get('containers', '[]'))

        crca_data = {
            'metrics': metrics,
            'containers': containers
        }
        crca_info = {
            'settings': get_settings(),
            'data': crca_data
        }
        crca_info_json = json.dumps(crca_info)
        try:
            response = requests.post(f'{settings.API_DATA_PROCESSING_URL}/preprocess_crca_data',
                                     files={'crca_file': crca_file}, data={'crca_info': crca_info_json})
            response_data = response.json()
            task_id = response_data.get('task_id')

            return render(request, 'config_app/waiting_page.html', {'task_id': task_id,
                                                                    'api_url': settings.API_CRCA_ANOMALY_DETECTION_URL,
                                                                    'task_type': 'crca'})

        except RequestException as e:
            messages.error(request, f'Failed to upload file: {str(e)}')
            return redirect('home')

    metrics = get_metrics()
    return render(request, 'config_app/anomaly_detection/crca/crca_upload_data.html', {'metrics': metrics})
