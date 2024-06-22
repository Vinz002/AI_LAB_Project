from uuid import uuid4

from flask import Flask, request, jsonify, send_from_directory, url_for
import os

from PIL import Image

from test_with_plate import predict_car_brand_and_license_plate

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set the result folder
RESULT_FOLDER = 'static'
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the upload folder exists
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


@app.route('/')
def home():
    return send_from_directory("templates", "main.html")


@app.route('/api', methods=['POST'])
def api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        print("Successfully received file", file_path)
        print("Generating prediction")

        predicted_brand, license_plate_text, result_image = predict_car_brand_and_license_plate(file_path)

        result_filename = str(uuid4()) + ".png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_image = Image.fromarray(result_image)
        result_image.save(result_path)

        print("Prediction finished")

        results = [
            {"id": "brand", "title": predicted_brand, "subtitle": "Car Brand", "icon": "mdi-car-hatchback"},
            {"id": "licensePlate", "title": license_plate_text, "subtitle": "License Plate",
             "icon": "mdi-alphabetical-variant"},
        ]

        return jsonify({
            'message': 'File successfully uploaded',
            'filename': url_for(RESULT_FOLDER, filename=result_filename),
            'results': results,
        })


if __name__ == '__main__':
    app.run(debug=True)
