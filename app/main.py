import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from utility.fileHelper import save_file,  get_all_files_from_directory
from utility.trainingHelper import start_training_model, get_response,get_dataset
from prometheus_client import Counter, Histogram
from prometheus_client.exposition import generate_latest

APP = Flask(__name__)
SWAGGER_URL='/swagger'
API_URL='/static/swagger.json'
swaggerui_blueprint= get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name':'Who-is-Dajana-TinyLllama-REST'
    }
)
APP.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Define Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP Request Duration')


@APP.route("/upload-file", methods=["POST"])
def upload_file():
     with REQUEST_DURATION.time():
        REQUEST_COUNT.labels(method='POST', endpoint='/upload-file').inc()
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        content = file.read().decode('utf-8')
        save_file(file.filename, content)
        get_dataset()
        return jsonify({'content': "File uploaded!"}), 200

@APP.route("/train-model", methods=["GET"])
def train_model():
     with REQUEST_DURATION.time():
        REQUEST_COUNT.labels(method='GET', endpoint='/train-model').inc()
        ##STEP 1: gather all uploaded files
        fileExist = get_all_files_from_directory()
        if(fileExist == False):
            return jsonify({'content': "File is not uploaded"}), 500

        ##STEP 2: start training model
        model = start_training_model()

        return jsonify({'content' : "Model is trained"}), 200

@APP.route("/get-reply", methods=["POST"])
def get_reply():
     with REQUEST_DURATION.time():
        REQUEST_COUNT.labels(method='POST', endpoint='/get-reply').inc()
        message = request.form.get('message')
        if(not message):
            return jsonify({'content': "Input is empty"}), 500

        answer = get_response(message)

        return jsonify({'content' : answer}), 200


@APP.route('/metrics', methods=["GET"])
def metrics():
    metrics_data = generate_latest()
    metrics_str = metrics_data.decode('utf-8')  # Decode bytes to string
    return jsonify({'content': metrics_str}), 200

if __name__ == '__main__':

    PORT = int(os.environ.get('PORT', 5000))
    CORS = CORS(APP)
    APP.run(host='0.0.0.0', port=PORT, debug=True)
