from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings


@api_view(['GET'])
def get_settings(request):
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
    return Response(exposed_settings)
