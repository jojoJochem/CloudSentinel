import requests
import json
from requests.exceptions import RequestException
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

from ..forms import create_dynamic_form


def get_available_models(api_url):
    try:
        response = requests.get(f'{api_url}/get_available_models')
        return response.json()
    except RequestException:
        return {}


def get_config(API_URL):
    try:
        config_url = f'{API_URL}/get_config'
        response = requests.get(config_url)
        return response.json()
    except RequestException:
        return {}


def update_config(request, API_URL, redirect_url):
    config_data = get_config(API_URL)
    ConfigForm = create_dynamic_form(config_data)
    if request.method == 'POST':
        form = ConfigForm(request.POST)
        if form.is_valid():
            try:
                requests.post(f'{API_URL}/update_config', json=form.cleaned_data)
                messages.success(request, 'Config updated successfully!')
                return redirect(redirect_url)
            except requests.RequestException as e:
                messages.error(request, f'Failed to update: {str(e)}')
    else:
        form = ConfigForm()
    return render(request, 'config_app/update_config.html', {'form': form, 'redirect_url': redirect_url})


def update_config_internal(request, redirect_url):
    config_path = 'monitoring_project/config.json'
    with open(config_path) as config_file:
        config_data = json.load(config_file)

    ConfigForm = create_dynamic_form(config_data)

    if request.method == 'POST':
        form = ConfigForm(request.POST)
        if form.is_valid():
            updated_config = {key: form.cleaned_data[key] for key in config_data.keys()}
            with open(config_path, 'w') as config_file:
                json.dump(updated_config, config_file, indent=4)
            return redirect(redirect_url)
    else:
        form = ConfigForm()

    return render(request, 'config_app/update_config.html', {'form': form, 'redirect_url': redirect_url})


def get_metrics():
    try:
        response = requests.get(f'{settings.API_DATA_INGESTION_URL}/get_metrics')
        return response.json()
    except RequestException:
        return {}


def get_pods(namespace):
    try:
        response = requests.post(f'{settings.API_DATA_INGESTION_URL}/get_pod_names', json={'namespace': namespace})
        return response.json()
    except RequestException:
        return []


def get_settings():
    exposed_settings = {
        'API_DATA_INGESTION_URL': settings.API_DATA_INGESTION_URL,
        'API_DATA_PROCESSING_URL': settings.API_DATA_PROCESSING_URL,
        'API_CRCA_ANOMALY_DETECTION_URL': settings.API_CRCA_ANOMALY_DETECTION_URL,
        'API_CGNN_ANOMALY_DETECTION_URL': settings.API_CGNN_ANOMALY_DETECTION_URL,
        'API_LEARNING_ADAPTATION_URL': settings.API_LEARNING_ADAPTATION_URL,
        'PROMETHEUS_URL': settings.PROMETHEUS_URL,
        'CLUSTER_NAMESPACE': settings.CLUSTER_NAMESPACE,
        'KUBE_CONFIG_PATH': settings.KUBE_CONFIG_PATH,
    }
    return exposed_settings
