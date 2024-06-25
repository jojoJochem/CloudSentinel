import requests
import logging
import pandas as pd
import json
from flask import Response
from config import get_config

logging.basicConfig(level=logging.INFO)


def collect_crca_data(crca_data):
    """
    Collects CRCA data and processes it for anomaly detection.

    Args:
        crca_data (dict): The CRCA data including pod names, metrics, time range, and settings.

    Returns:
        Response: Flask response object containing the result from the CRCA preprocessing API.
    """
    dataframe = fetch_metrics(crca_data['data']['crca_pods'], crca_data['data']['metrics'],
                              crca_data['data']['start_time'], crca_data['data']['end_time'],
                              crca_data['settings']['PROMETHEUS_URL'], crca_data['data']['step'])
    crca_file = {'crca_file': dataframe.to_csv(header=False, index=False)}
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
    """
    Fetches metrics from Prometheus for the specified pods and metrics.

    Args:
        pod_names (list): List of pod names to fetch metrics for.
        metric_queries (list): List of metric queries to execute.
        start_time (int): Start time for the metric data in Unix timestamp.
        end_time (int): End time for the metric data in Unix timestamp.
        url (str): URL of the Prometheus server.
        step (int, optional): Step interval for fetching metrics in seconds. Defaults to 60.

    Returns:
        DataFrame: A pandas DataFrame containing the fetched metrics.
    """
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
            if results:
                timestamps = [pd.Timestamp.fromtimestamp(int(value[0])) for value in results[0]['values']]
                measures = [float(value[1]) for value in results[0]['values']]
                metric_df = pd.DataFrame({
                    'timestamp': timestamps,
                    f"{pod}_{metric}": measures
                })
                master_df = pd.merge(master_df, metric_df, on='timestamp', how='left')

    # Fill missing data
    master_df.ffill(inplace=True)
    master_df.bfill(inplace=True)
    master_df.fillna(0, inplace=True)
    master_df = master_df.drop('timestamp', axis=1)
    return master_df


def query_prometheus(query, start_time, end_time, url, step=5):
    """
    Queries Prometheus for metric data.

    Args:
        query (str): Prometheus query string.
        start_time (int): Start time for the query in Unix timestamp.
        end_time (int): End time for the query in Unix timestamp.
        url (str): URL of the Prometheus server.
        step (int, optional): Step interval for the query in seconds. Defaults to 5.

    Returns:
        list: A list of results from Prometheus.

    Raises:
        HTTPError: If an HTTP error occurs during the request.
        Exception: If any other error occurs during the request.
    """
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
