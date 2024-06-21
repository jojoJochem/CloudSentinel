import requests
import logging
import pandas as pd
import time
import json
from flask import Response

from config import get_config

logging.basicConfig(level=logging.INFO)


def get_metrics_test():
    pod_names = ['kube-apiserver-minikube', 'kube-controller-manager-minikube', 'kube-proxy-frt49', 'kube-scheduler-minikube']
    metric_queries = ['cpu_usage_pods', 'memory_usage_pods', 'network_io_bytes_received', 'network_io_bytes_transmitted', 'pod_read_bytes', 'pod_write_bytes']
    end_time = int(time.time())
    start_time = end_time - 180
    url = 'http://127.0.0.1:55725'
    step = 5

    return fetch_metrics(pod_names, metric_queries, start_time, end_time, url, step)


def collect_crca_data(crca_data):
    # dataframe = fetch_metrics(crca_data['data']['crca_pods'], crca_data['data']['metrics'],
    #                           crca_data['data']['start_time'], crca_data['data']['end_time'],
    #                           crca_data['settings']['PROMETHEUS_URL'], crca_data['data']['step'])
    # crca_file = {'crca_file': dataframe.to_csv(header=False, index=False)}
    crca_file = {'crca_file': open('data_full.csv', 'rb')}
    crca_info_json = json.dumps(crca_data)
    response = requests.post(f"{crca_data['settings']['API_DATA_PROCESSING_URL']}/preprocess_crca_data",
                             files=crca_file, data={'crca_info': crca_info_json})
    flask_response = Response(
        response.content,
        status=response.status_code,
        content_type=response.headers['Content-Type']
    )

    return flask_response


def fetch_metrics(pod_names, metric_queries, start_time, end_time, url, step=60):
    all_timestamps = pd.date_range(
        start=pd.Timestamp.fromtimestamp(start_time),
        end=pd.Timestamp.fromtimestamp(end_time),
        freq=f'{step}s'
    )
    master_df = pd.DataFrame(all_timestamps, columns=['timestamp'])
    metrics_config = get_config()
    for pod in pod_names:
        for metric in metric_queries:
            query = metrics_config[metric]['query'].format(pod=pod)
            results = query_prometheus(query, start_time, end_time, url, step)
            logging.info(f"{pod} {metric}")
            if results:
                timestamps = [pd.Timestamp.fromtimestamp(int(value[0])) for value in results[0]['values']]
                measures = [float(value[1]) for value in results[0]['values']]
                metric_df = pd.DataFrame({
                    'timestamp': timestamps,
                    f"{pod}_{metric}": measures
                })
                master_df = pd.merge(master_df, metric_df, on='timestamp', how='left')
                logging.info("\tQuery successful, data merged.")

    # Fill missing data
    master_df.ffill(inplace=True)
    master_df.bfill(inplace=True)
    master_df.fillna(0, inplace=True)
    master_df = master_df.drop('timestamp', axis=1)
    return master_df


def query_prometheus(query, start_time, end_time, url, step=5):
    url = f"{url}/api/v1/query_range"
    params = {'query': query, 'start': start_time, 'end': end_time, 'step': f'{step}s'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()['data']['result']
        return results
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred: {err}")
    except Exception as err:
        logging.error(f"Other error occurred: {err}")


if __name__ == '__main__':
    get_metrics_test()
