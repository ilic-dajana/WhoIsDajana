import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from utility.fileHelper import save_file,  get_all_files_from_directory
from utility.trainingHelper import start_training_model, get_response


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

@APP.route("/upload-file", methods=["POST"])
def upload_file():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    content = file.read().decode('utf-8')
    save_file(file.filename, content)
    return jsonify({'content': "File uploaded!"}), 200

@APP.route("/train-model", methods=["GET"])
def train_model():
    ##STEP 1: gather all uploaded files
    fileExist = get_all_files_from_directory()
    if(fileExist == False):
        return jsonify({'content': "File is not uploaded"}), 500

    ##STEP 2: start training model
    model = start_training_model()

    return jsonify({'content' : "Model is trained"}), 200

@APP.route("/get-reply", methods=["GET"])
def get_reply():
    message = request.form.get('message')

    if(not message):
        return jsonify({'content': "Input is empty"}), 500
    
    answer = get_response(message)

    return jsonify({'content' : answer}), 200




if __name__ == '__main__':  

    PORT = int(os.environ.get('PORT', 5000))
    CORS = CORS(APP)
    APP.run(host='0.0.0.0', port=PORT, debug=True)
   