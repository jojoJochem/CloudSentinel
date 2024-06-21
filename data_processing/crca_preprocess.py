from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
from flask import Response


def handle_crca_request(crca_file, crca_info):
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
    df = pd.read_csv(csv, header=None)
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)
    return pd.DataFrame(df_normalized).to_csv(index=False, header=False)

    # df = pd.read_csv(csv)
    # scaler = MinMaxScaler()
    # feature_columns = [col for col in df.columns if col != 'timestamp']
    # df[feature_columns] = scaler.fit_transform(df[feature_columns])
    # return df.to_csv(index=False)
