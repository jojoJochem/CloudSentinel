import requests
import json
from requests.exceptions import RequestException
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

from ..forms import create_dynamic_form


def get_available_models(api_url):
    """
    Retrieves the available models from the given API URL.

    Args:
        api_url (str): The URL of the API to fetch available models from.

    Returns:
        dict: A dictionary of available models, or an empty dictionary on failure.
    """
    try:
        response = requests.get(f'{api_url}/get_available_models')
        response.raise_for_status()
        return response.json()
    except RequestException:
        return {}


def get_config(API_URL):
    """
    Retrieves the configuration from the given API URL.

    Args:
        API_URL (str): The URL of the API to fetch the configuration from.

    Returns:
        dict: A dictionary of the configuration, or an empty dictionary on failure.
    """
    try:
        config_url = f'{API_URL}/get_config'
        response = requests.get(config_url)
        response.raise_for_status()
        return response.json()
    except RequestException:
        return {}


def update_config(request, API_URL, redirect_url):
    """
    Updates the configuration using data from the form and the given API URL.

    Args:
        request (HttpRequest): The request object.
        API_URL (str): The URL of the API to update the configuration.
        redirect_url (str): The URL to redirect to after a successful update.

    Returns:
        HttpResponse: The rendered update configuration page or a redirect response on success.
    """
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
    """
    Updates the internal configuration stored in a JSON file.

    Args:
        request (HttpRequest): The request object.
        redirect_url (str): The URL to redirect to after a successful update.

    Returns:
        HttpResponse: The rendered update configuration page or a redirect response on success.
    """
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
            messages.success(request, 'Config updated successfully!')
            return redirect(redirect_url)
    else:
        form = ConfigForm()

    return render(request, 'config_app/update_config.html', {'form': form, 'redirect_url': redirect_url})


def get_metrics():
    """
    Retrieves the metrics from the data ingestion API.

    Returns:
        dict: A dictionary of metrics, or an empty dictionary on failure.
    """
    try:
        response = requests.get(f'{settings.API_DATA_INGESTION_URL}/get_metrics')
        response.raise_for_status()
        return response.json()
    except RequestException:
        return {}


def get_pods(namespace):
    """
    Retrieves the pod names for the given namespace.

    Args:
        namespace (str): The Kubernetes namespace to fetch pod names from.

    Returns:
        list: A list of pod names, or an empty list on failure.
    """
    try:
        response = requests.post(f'{settings.API_DATA_INGESTION_URL}/get_pod_names', json={'namespace': namespace})
        response.raise_for_status()
        return response.json()
    except RequestException:
        return []


def get_settings():
    """
    Exposes the settings needed for various configurations and URLs.

    Returns:
        dict: A dictionary of exposed settings.
    """
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
