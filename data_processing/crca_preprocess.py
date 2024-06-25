from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
from flask import Response

def handle_crca_request(crca_file, crca_info):
    """
    Handles the request for CRCA anomaly detection.

    Args:
        crca_file (FileStorage): The CRCA data file uploaded by the user.
        crca_info (dict): Information about the CRCA including settings.

    Returns:
        Response: Flask response object containing the result from the CRCA anomaly detection API.
    """
    crca_file_processed = normalize_data(crca_file)
    crca_info_json = json.dumps(crca_info)
    response = requests.post(f"{crca_info['settings']['API_CRCA_ANOMALY_DETECTION_URL']}/crca",
                             files={'crca_file': crca_file_processed}, data={'crca_info': crca_info_json})

    flask_response = Response(
        response.content,
        status=response.status_code,
        content_type=response.headers['Content-Type']
    )

    return flask_response

def normalize_data(csv):
    """
    Normalizes the data in the CSV file using MinMaxScaler.

    Args:
        csv (FileStorage): The CSV file containing the data to be normalized.

    Returns:
        str: A CSV formatted string of the normalized data.
    """
    df = pd.read_csv(csv, header=None)
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)
    return pd.DataFrame(df_normalized).to_csv(index=False, header=False)
