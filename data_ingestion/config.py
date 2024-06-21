# config.py
import json

config_file_path = "metric_config.json"


def get_config():
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    return config


def set_config(new_config=None):
    config = get_config()
    if new_config:
        config.update(new_config)
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)


def set_initial_metric_config():
    config = {
        "cpu_usage_pods": {
            "query": 'sum(rate(container_cpu_usage_seconds_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "CPU usage"
        },
        "memory_usage_pods": {
            "query": 'sum(container_memory_usage_bytes{{pod="{pod}"}}) by (pod)',
            "info": "Memory usage"
        },
        "cpu_limit_pods": {
            "query": 'sum(kube_pod_container_resource_limits_cpu_cores{{pod="{pod}"}}) by (pod)',
            "info": "CPU limit"
        },
        "memory_limit_pods": {
            "query": 'sum(kube_pod_container_resource_limits_memory_bytes{{pod="{pod}"}}) by (pod)',
            "info": "Memory limit"
        },
        "cpu_requests_pods": {
            "query": 'sum(kube_pod_container_resource_requests_cpu_cores{{pod="{pod}"}}) by (pod)',
            "info": "CPU requests"
        },
        "memory_requests_pods": {
            "query": 'sum(kube_pod_container_resource_requests_memory_bytes{{pod="{pod}"}}) by (pod)',
            "info": "Memory requests"
        },
        "pod_restarts": {
            "query": 'sum(rate(kube_pod_container_status_restarts_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Number of restarts"
        },
        "pod_status_running": {
            "query": 'sum(kube_pod_status_phase{{pod="{pod}", phase="Running"}}) by (pod)',
            "info": "Number of pods in running state"
        },
        "pod_status_pending": {
            "query": 'sum(kube_pod_status_phase{{pod="{pod}", phase="Pending"}}) by (pod)',
            "info": "Number of pods in pending state"
        },
        "pod_status_failed": {
            "query": 'sum(kube_pod_status_phase{{pod="{pod}", phase="Failed"}}) by (pod)',
            "info": "Number of pods in failed state"
        },
        "pod_status_succeeded": {
            "query": 'sum(kube_pod_status_phase{{pod="{pod}", phase="Succeeded"}}) by (pod)',
            "info": "Number of pods in succeeded state"
        },
        "disk_io_operations": {
            "query": 'sum(rate(container_fs_reads_bytes_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Disk I/O operations"
        },
        "network_io_bytes_received": {
            "query": 'sum(rate(container_network_receive_bytes_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Network bytes received by pods"
        },
        "network_io_bytes_transmitted": {
            "query": 'sum(rate(container_network_transmit_bytes_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Network bytes transmitted by pods"
        },
        "pod_evictions": {
            "query": 'sum(rate(kube_pod_evictions_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Number of pod evictions"
        },
        "pod_uptime": {
            "query": 'time() - kube_pod_start_time{{pod="{pod}"}}',
            "info": "Uptime of pod"
        },
        "pods_by_node": {
            "query": 'sum(kube_pod_info{{pod="{pod}"}}) by (node)',
            "info": "Number of pods by node"
        },
        "pod_cpu_throttling": {
            "query": 'sum(rate(container_cpu_cfs_throttled_seconds_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "CPU throttling"
        },
        "pod_oom_kills": {
            "query": 'sum(rate(kube_pod_container_status_oomkilled{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Out Of Memory kills"
        },
        "pod_read_bytes": {
            "query": 'sum(rate(container_fs_reads_bytes_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Read bytes"
        },
        "pod_write_bytes": {
            "query": 'sum(rate(container_fs_writes_bytes_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Write bytes"
        },
        "pod_log_size": {
            "query": 'sum(container_fs_usage_bytes{{pod="{pod}", device="log"}}) by (pod)',
            "info": "Log size of pod"
        },
        "pod_fs_inodes_free": {
            "query": 'sum(container_fs_inodes_free{{pod="{pod}"}}) by (pod)',
            "info": "Free inodes"
        },
        "pod_fs_inodes_total": {
            "query": 'sum(container_fs_inodes_total{{pod="{pod}"}}) by (pod)',
            "info": "Total inodes"
        },
        "pod_fs_inodes_used": {
            "query": 'sum(container_fs_inodes_used{{pod="{pod}"}}) by (pod)',
            "info": "Used inodes"
        },
        "pod_disk_space_available": {
            "query": 'sum(container_fs_available_bytes{{pod="{pod}"}}) by (pod)',
            "info": "Available disk space"
        },
        "pod_network_errors": {
            "query": 'sum(rate(container_network_receive_errors_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Network errors"
        },
        "pod_network_dropped_packets": {
            "query": 'sum(rate(container_network_receive_packets_dropped_total{{pod="{pod}"}}[1m])) by (pod)',
            "info": "Network dropped packets"
        }
    }

    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=2)



# def set_initial_metric_config():
#     config = {
#         "cpu_usage_pods": {
#             "query": 'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "CPU usage"
#         },
#         "memory_usage_pods": {
#             "query": 'sum(container_memory_usage_bytes{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "Memory usage"
#         },
#         "cpu_limit_pods": {
#             "query": 'sum(kube_pod_container_resource_limits_cpu_cores{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "CPU limit"
#         },
#         "memory_limit_pods": {
#             "query": 'sum(kube_pod_container_resource_limits_memory_bytes{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "Memory limit"
#         },
#         "cpu_requests_pods": {
#             "query": 'sum(kube_pod_container_resource_requests_cpu_cores{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "CPU requests"
#         },
#         "memory_requests_pods": {
#             "query": 'sum(kube_pod_container_resource_requests_memory_bytes{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "Memory requests"
#         },
#         "pod_restarts": {
#             "query": 'sum(rate(kube_pod_container_status_restarts_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Number of restarts"
#         },
#         "pod_status_running": {
#             "query": 'sum(kube_pod_status_phase{{namespace="{namespace}", pod="{pod}", phase="Running"}}) by (pod)',
#             "info": "Number of pods in running state"
#         },
#         "pod_status_pending": {
#             "query": 'sum(kube_pod_status_phase{{namespace="{namespace}", pod="{pod}", phase="Pending"}}) by (pod)',
#             "info": "Number of pods in pending state"
#         },
#         "pod_status_failed": {
#             "query": 'sum(kube_pod_status_phase{{namespace="{namespace}", pod="{pod}", phase="Failed"}}) by (pod)',
#             "info": "Number of pods in failed state"
#         },
#         "pod_status_succeeded": {
#             "query": 'sum(kube_pod_status_phase{{namespace="{namespace}", pod="{pod}", phase="Succeeded"}}) by (pod)',
#             "info": "Number of pods in succeeded state"
#         },
#         "disk_io_operations": {
#             "query": 'sum(rate(container_fs_reads_bytes_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Disk I/O operations"
#         },
#         "network_io_bytes_received": {
#             "query": 'sum(rate(container_network_receive_bytes_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Network bytes received by pods"
#         },
#         "network_io_bytes_transmitted": {
#             "query": 'sum(rate(container_network_transmit_bytes_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Network bytes transmitted by pods"
#         },
#         "pod_evictions": {
#             "query": 'sum(rate(kube_pod_evictions_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Number of pod evictions"
#         },
#         "pod_uptime": {
#             "query": 'time() - kube_pod_start_time{{namespace="{namespace}", pod="{pod}"}}',
#             "info": "Uptime of pod"
#         },
#         "pods_by_node": {
#             "query": 'sum(kube_pod_info{{namespace="{namespace}", pod="{pod}"}}) by (node)',
#             "info": "Number of pods by node"
#         },
#         "pod_cpu_throttling": {
#             "query": 'sum(rate(container_cpu_cfs_throttled_seconds_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "CPU throttling"
#         },
#         "pod_oom_kills": {
#             "query": 'sum(rate(kube_pod_container_status_oomkilled{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Out Of Memory kills"
#         },
#         "pod_read_bytes": {
#             "query": 'sum(rate(container_fs_reads_bytes_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Read bytes"
#         },
#         "pod_write_bytes": {
#             "query": 'sum(rate(container_fs_writes_bytes_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Write bytes"
#         },
#         "pod_log_size": {
#             "query": 'sum(container_fs_usage_bytes{{namespace="{namespace}", pod="{pod}", device="log"}}) by (pod)',
#             "info": "Log size of pod"
#         },
#         "pod_fs_inodes_free": {
#             "query": 'sum(container_fs_inodes_free{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "Free inodes"
#         },
#         "pod_fs_inodes_total": {
#             "query": 'sum(container_fs_inodes_total{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "Total inodes"
#         },
#         "pod_fs_inodes_used": {
#             "query": 'sum(container_fs_inodes_used{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "Used inodes"
#         },
#         "pod_disk_space_available": {
#             "query": 'sum(container_fs_available_bytes{{namespace="{namespace}", pod="{pod}"}}) by (pod)',
#             "info": "Available disk space"
#         },
#         "pod_network_errors": {
#             "query": 'sum(rate(container_network_receive_errors_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Network errors"
#         },
#         "pod_network_dropped_packets": {
#             "query": 'sum(rate(container_network_receive_packets_dropped_total{{namespace="{namespace}", pod="{pod}"}}[1m])) by (pod)',
#             "info": "Network dropped packets"
#         }
#     }

#     with open(config_file_path, 'w') as file:
#         json.dump(config, file)
