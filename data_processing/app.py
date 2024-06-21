# app.py
from flask import Flask, request, jsonify
import logging
import json
import traceback

from cgnn_preprocess import handle_cgnn_request, handle_cgnn_train_request
from crca_preprocess import handle_crca_request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/preprocess_cgnn_train_data', methods=['POST'])
def preprocess_cgnn_train_data():
    try:
        train_data = request.files
        train_info_json = request.form.get('train_info')
        train_info = json.loads(train_info_json)
        response = handle_cgnn_train_request(train_data, train_info)
        return response
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/preprocess_cgnn_data', methods=['POST'])
def preprocess_cgnn_data():
    try:
        test_data = request.files['test_array']
        test_info_json = request.form.get('test_info')
        test_info = json.loads(test_info_json)
        response = handle_cgnn_request(test_data, test_info)
        return response
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/preprocess_crca_data', methods=['POST'])
def preprocess_crca_data():
    try:
        crca_file = request.files['crca_file']
        crca_info_json = request.form.get('crca_info')
        crca_info = json.loads(crca_info_json)
        response = handle_crca_request(crca_file, crca_info)
        return response
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
