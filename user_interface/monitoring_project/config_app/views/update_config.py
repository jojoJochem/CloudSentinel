from .utils import update_config, update_config_internal
from django.conf import settings


def config(request):
    return update_config_internal(request, 'home')


def config_cgnn(request):
    return update_config(request, settings.API_CGNN_ANOMALY_DETECTION_URL, 'perform_anomaly_detection_cgnn')


def config_crca(request):
    return update_config(request, settings.API_CRCA_ANOMALY_DETECTION_URL, 'perform_anomaly_detection_crca')


def config_cgnn_train(request):
    return update_config(request, settings.API_LEARNING_ADAPTATION_URL, 'train_cgnn')
