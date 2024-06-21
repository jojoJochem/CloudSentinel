import numpy as np
import pandas as pd
import requests
import json
from flask import Response

from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler


def handle_cgnn_request(test_data, test_info):
    test_array, _, _ = load_data(test_data)
    test_data_processed = get_data(test_array, test_array.shape[1])
    test_files = {'test_array': test_data_processed}
    test_info_json = json.dumps(test_info)
    response = requests.post(f"{test_info['settings']['API_CGNN_ANOMALY_DETECTION_URL']}/detect_anomalies",
                             files=test_files, data={'test_info': test_info_json})

    flask_response = Response(
        response.content,
        status=response.status_code,
        content_type=response.headers['Content-Type']
    )
    return flask_response


def handle_cgnn_train_request(cgnn_data, train_info):
    test_array, train_array, anomaly_label_array = load_data(cgnn_data['test_array'],
                                                             cgnn_data['train_array'],
                                                             cgnn_data['anomaly_label_array'],
                                                             train_info['data']['anomaly_sequence'])
    train_data, test_data, anomaly_data = get_data(test_array, test_array.shape[1],
                                                   train_array, anomaly_label_array)
    train_files = {
        'train_array': train_data,
        'test_array': test_data,
        'anomaly_label_array': anomaly_data
    }
    train_info_json = json.dumps(train_info)
    response = requests.post(f"{train_info['settings']['API_LEARNING_ADAPTATION_URL']}/cgnn_train_model",
                             files=train_files, data={'train_info': train_info_json})

    flask_response = Response(
        response.content,
        status=response.status_code,
        content_type=response.headers['Content-Type']
    )
    return flask_response


def load_data(test_data, train_data=None, anomaly_label=None, anomaly_sequence="False"):
    test_df = pd.read_csv(test_data, header=None)
    test_array = test_df.to_numpy(dtype=np.float32)
    if train_data is not None:
        train_df = pd.read_csv(train_data, header=None)
        train_array = train_df.to_numpy(dtype=np.float32)
    else:
        train_array = None
    if anomaly_label is not None:
        if anomaly_sequence == "True":
            anomalies = literal_eval(anomaly_label, header=None)
            length = len(test_array)
            label = np.zeros([length], dtype=np.float32)
            for anomaly in anomalies:
                label[anomaly[0]: anomaly[1] + 1] = 1.0
            anomaly_label_array = np.asarray(label)
        else:
            anomaly_label_df = pd.read_csv(anomaly_label, header=None)
            anomaly_label_array = anomaly_label_df.to_numpy(dtype=np.float32)
    else:
        anomaly_label_array = None
    return test_array, train_array, anomaly_label_array


def get_data(test_array, data_dim, train_array=None, anomaly_label_array=None):
    test_data = test_array.reshape((-1, data_dim))
    if train_array is not None:
        train_data = train_array.reshape((-1, data_dim))
    if anomaly_label_array is not None:
        anomaly_data = anomaly_label_array.reshape((-1))

    if anomaly_label_array is not None:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)
        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        print("test set label shape: ", None if anomaly_data is None else anomaly_data.shape)
        return pd.DataFrame(train_data).to_csv(index=False, header=False), pd.DataFrame(test_data).to_csv(index=False, header=False), pd.DataFrame(anomaly_data).to_csv(index=False, header=False)
    else:
        test_data, _ = normalize_data(test_data, scaler=None)
        print("test set shape: ", test_data.shape)
        return pd.DataFrame(test_data).to_csv(index=False, header=False)


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)

    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler
