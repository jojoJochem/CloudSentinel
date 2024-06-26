import requests
from requests.exceptions import RequestException
from django.shortcuts import render
from ..forms import MonitoringForm, create_dynamic_form
from .utils import get_pods, get_available_models, get_config, get_settings
from django.conf import settings
from django.http import JsonResponse
import json


def monitoring_home(request):
    """
    Renders the home page for monitoring, loading the Kubernetes configuration and pod names.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered monitoring home page with pod names.
    """
    load_kube_config()
    pod_names = get_pods(settings.CLUSTER_NAMESPACE)
    return render(request, 'config_app/monitoring/monitoring_home.html', {'pod_names': pod_names})


def monitoring(request):
    """
    Handles the monitoring setup, including form handling and initiating monitoring.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered monitoring setup page with form and models data.
        JsonResponse: A JSON response indicating success or failure of the monitoring initiation.
    """
    load_kube_config()
    config_data = get_config(settings.API_CRCA_ANOMALY_DETECTION_URL)
    crca_form = create_dynamic_form(config_data)()

    if request.method == 'POST':
        pods = get_pods(settings.CLUSTER_NAMESPACE)
        pod_choices = [(pod, pod) for pod in pods]
        model_choices = get_available_models(settings.API_CGNN_ANOMALY_DETECTION_URL)
        form = MonitoringForm(request.POST)
        form.fields['containers'].choices = pod_choices
        form.fields['model'].choices = [(model, model) for model in model_choices]
        form.fields['crca_pods'].choices = pod_choices

        if form.is_valid():
            # Collect data from the form
            containers = form.cleaned_data['containers']
            selected_model = form.cleaned_data['model']
            data_interval = form.cleaned_data['data_interval']
            duration = form.cleaned_data['duration']
            test_interval = form.cleaned_data['test_interval']
            crca_threshold = form.cleaned_data['crca_threshold']
            crca_pods = form.cleaned_data['crca_pods']
            crca_config = json.loads(request.POST.get('crca_config_data', '{}'))

            # Prepare the monitoring data
            monitor_data = {
                'metrics': model_choices[selected_model]['model_params']['metrics'],
                'containers': containers,
                'data_interval': data_interval,
                'duration': duration,
                'test_interval': test_interval,
                'model': selected_model,
                'crca_threshold': crca_threshold,
                'crca_pods': crca_pods,
                'crca_config': crca_config
            }
            monitor_info = {
                'settings': get_settings(),
                'data': monitor_data
            }
            monitor_info_json = json.dumps(monitor_info)

            # Send the monitoring data to the API
            try:
                requests.post(f'{settings.API_DATA_INGESTION_URL}/start_monitoring', data={'monitor_info': monitor_info_json})
                return JsonResponse({'status': 'success'})
            except RequestException as e:
                return JsonResponse({'status': 'error', 'message': str(e)})

    # Handle the GET request and initialize the form
    pods = get_pods(settings.CLUSTER_NAMESPACE)
    pod_choices = [(pod, pod) for pod in pods]
    model_choices = get_available_models(settings.API_CGNN_ANOMALY_DETECTION_URL)

    form = MonitoringForm()
    form.fields['containers'].choices = pod_choices
    form.fields['model'].choices = [(model, model) for model in model_choices]
    form.fields['crca_pods'].choices = pod_choices

    return render(request, 'config_app/monitoring/monitoring_setup.html', {'form': form, 'models': model_choices, 'crca_form': crca_form})


def monitoring_overview(request):
    """
    Renders the monitoring overview dashboard.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered monitoring dashboard page.
    """
    return render(request, 'config_app/monitoring/monitoring_dashboard.html',
                  {'flask_url': settings.API_CGNN_ANOMALY_DETECTION_URL, 'monitoring_id': 234})


def load_kube_config():
    """
    Loads the Kubernetes configuration.

    Returns:
        JsonResponse: A JSON response indicating success or failure of the operation.
    """
    try:
        requests.post(f'{settings.API_DATA_INGESTION_URL}/load_kube_config', json={'kube_config_path': settings.KUBE_CONFIG_PATH})
        return JsonResponse({'status': 'success'})
    except RequestException as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


def task_manager(request):
    """
    Renders the task manager page for monitoring.

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered task manager page.
    """
    return render(request, 'config_app/monitoring/monitoring_task_manager.html',
                  {'data_ingestion_url': settings.API_DATA_INGESTION_URL,
                   'cgnn_url': settings.API_CGNN_ANOMALY_DETECTION_URL,
                   'crca_url': settings.API_CRCA_ANOMALY_DETECTION_URL})
