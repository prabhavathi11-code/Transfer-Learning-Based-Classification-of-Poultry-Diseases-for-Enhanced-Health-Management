from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import webbrowser
from threading import Timer

app = Flask(__name__)
model = load_model("healthy_vs_rotten.h5")
disease_names = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result = disease_names[np.argmax(prediction)]

        return render_template('predict.html', prediction=result, image_path='/' + file_path)
    return render_template('predict.html')

def open_browser():
    webbrowser.open_new("http://localhost:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)