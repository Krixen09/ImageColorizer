from flask import Flask, render_template, request, jsonify, send_from_directory
from main import model, TestDataset, predict_outputs, save_outputs
import os
import uuid

app = Flask(__name__)

# Define the upload folder and the static folder
UPLOAD_FOLDER = 'uploads'
COLORIZED_FOLDER = 'colorized_images'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COLORIZED_FOLDER'] = COLORIZED_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})

        # Get the uploaded file
        uploaded_file = request.files['image']

        # If the user does not select a file, browser also
        # submit an empty part without filename
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'})

        if uploaded_file:
            # Generate a unique filename using UUID
            filename = str(uuid.uuid4()) + '.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the uploaded file
            uploaded_file.save(file_path)

            # Perform colorization using main.py
            colorized_images = predict_outputs(model, TestDataset(file_path))

            # Save colorized images to files
            save_outputs(colorized_images, app.config['COLORIZED_FOLDER'])

            # Get the file name of the first colorized image
            colorized_image_file = list(colorized_images.keys())[0]
            colorized_image_name = os.path.basename(colorized_image_file)

            # Construct the path to the colorized image
            colorized_image_path = os.path.join(app.config['COLORIZED_FOLDER'], colorized_image_name)

            # Return the path to the colorized image
            return jsonify({'colorized_image_path': colorized_image_path})

@app.route('/colorized_images/<filename>')
def serve_colorized_image(filename):
    return send_from_directory(app.config['COLORIZED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
