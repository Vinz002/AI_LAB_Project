from flask import Flask, request, jsonify, send_from_directory
import os

from test_with_plate import predict_car_brand_and_license_plate

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


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

        print("Prediction finished")

        results = [
            {"title": predicted_brand, "subtitle": "Car Brand", "icon": "mdi-car-hatchback"},
            {"title": license_plate_text, "subtitle": "License Plate", "icon": "mdi-alphabetical-variant"},
        ]

        return jsonify({'message': 'File successfully uploaded', 'filename': file.filename, 'results': results})


if __name__ == '__main__':
    app.run(debug=True)
