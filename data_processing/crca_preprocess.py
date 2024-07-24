from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
from flask import Response
from chardet import detect

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


def detect_encoding(file):
    raw_data = file.read()
    result = detect(raw_data)
    file.seek(0)  # Reset file pointer to the beginning
    return result['encoding']


def has_header(df):
    return any(isinstance(col, str) for col in df.columns)


def normalize_data(file):
    """
    Normalizes the data in the CSV or PKL file using MinMaxScaler.

    Args:
        file (FileStorage): The file containing the data to be normalized.

    Returns:
        str: A CSV formatted string of the normalized data.
    """
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    if file_extension not in ['csv', 'pkl', 'txt']:
        raise ValueError("Unsupported file type. Only CSV, PKL, and TXT files are supported.")

    encoding = detect_encoding(file)

    if file_extension == 'csv' or file_extension == 'txt':
        # Try to read the first row to check for a header
        first_row = pd.read_csv(file, nrows=1, encoding=encoding)
        file.seek(0)  # Reset file pointer to the beginning

        if has_header(first_row):
            # The first row contains headers
            df = pd.read_csv(file, header=0, encoding=encoding)
        else:
            # The first row does not contain headers
            df = pd.read_csv(file, header=None, encoding=encoding)
    elif file_extension == 'pkl':
        df = pd.read_pickle(file)
        if has_header(df):
            df.columns = range(df.shape[1])  # Remove the header by resetting column names

    print(df)
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)
    return pd.DataFrame(df_normalized).to_csv(index=False, header=False)
