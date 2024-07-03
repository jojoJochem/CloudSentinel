from django.urls import path
from .views.home import home
from .views.update_config import config, config_cgnn, config_crca, config_cgnn_train
from .views.training import train_algorithms, train_cgnn, upload_cgnn_train_data, cgnn_train_data
from .views.anomaly_detection import (home_anomaly_detection, perform_anomaly_detection_cgnn, perform_anomaly_detection_crca,
                                      select_crca_file, chosen_crca, upload_cgnn_data, upload_crca_data)
from .views.monitoring import monitoring, monitoring_home, monitoring_overview, task_manager
from .views.api import get_settings
from .views.task_results import (get_results, training_result, fetch_cgnn_results, fetch_crca_task_details,
                                 fetch_active_tasks, stop_task, fetch_results, delete_result, check_status)

urlpatterns = [
    path('', home, name='home'),

    path('config/', config, name='config'),
    path('config-cgnn/', config_cgnn, name='config_cgnn'),
    path('config-crca/', config_crca, name='config_crca'),
    path('config-cgnn-train/', config_cgnn_train, name='config_cgnn_train'),
    path('results/<task_id>/<task_type>', get_results, name='get_results'),

    path('train-algorithms/', train_algorithms, name='train_algorithms'),
    path('train-cgnn/', train_cgnn, name='train_cgnn'),
    path('upload-cgnn-train-data/', upload_cgnn_train_data, name='upload_cgnn_train_data'),
    path('cgnn-train-data/', cgnn_train_data, name='cgnn_train_data'),
    path('training-result', training_result, name='training_result'),

    path('home-anomaly-detection/', home_anomaly_detection, name='home_anomaly_detection'),
    path('perform-anomaly-detection-crca/', perform_anomaly_detection_crca, name='perform_anomaly_detection_crca'),
    path('select-crca-file/', select_crca_file, name='select_crca_file'),
    path('chosen-crca/', chosen_crca, name='chosen_crca'),
    path('upload-crca-data/', upload_crca_data, name='upload_crca_data'),

    path('perform-anomaly-detection-cgnn/', perform_anomaly_detection_cgnn, name='perform_anomaly_detection_cgnn'),
    path('upload-cgnn-data/', upload_cgnn_data, name='upload_cgnn_data'),

    path('monitoring-home/', monitoring_home, name='monitoring_home'),
    path('monitoring/', monitoring, name='monitoring'),
    path('monitoring-overview/', monitoring_overview, name='monitoring_overview'),
    path('task_manager/', task_manager, name='task_manager'),

    path('api/settings/', get_settings, name='get_settings'),
    path('check_status/<task_id>/<task_type>', check_status, name='check_status'),

    path('fetch_cgnn_results/', fetch_cgnn_results, name='fetch_cgnn_results'),
    path('fetch_crca_task_details/', fetch_crca_task_details, name='fetch_crca_task_details'),

    path('fetch_active_tasks/', fetch_active_tasks, name='fetch_active_tasks'),
    path('stop-task/<taskId>/', stop_task, name='stop_task'),
    path('fetch_results/', fetch_results, name='fetch_results'),
    path('delete-result/<taskId>/', delete_result, name='delete_result'),
]
