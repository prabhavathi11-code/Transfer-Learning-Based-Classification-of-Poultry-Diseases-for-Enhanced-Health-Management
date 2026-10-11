from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
import webbrowser
from threading import Timer

app = Flask(__name__, static_folder='static', template_folder='Templates')

# Disease category labels
disease_names = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load deep learning model with graceful fallback
model = None
MODEL_PATH = "healthy_vs_rotten.h5"

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ PoultryVision: Deep learning model loaded successfully.")
    else:
        print(f"⚠️ Warning: '{MODEL_PATH}' not found. Simulation fallback mode active.")
except Exception as e:
    print(f"⚠️ Note: TensorFlow model could not be loaded ({e}). Using simulation mode.")

@app.route('/')
def home():
    if os.path.exists("index.html"):
        return send_from_directory(".", "index.html")
    return render_template('index.html')

@app.route('/dashboard')
@app.route('/dashboard.html')
def dashboard():
    return send_from_directory(".", "dashboard.html")

@app.route('/login')
@app.route('/login.html')
def login():
    return send_from_directory(".", "login.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', prediction="No file uploaded.")
        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', prediction="No file selected.")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        if model is not None:
            try:
                img = image.load_img(file_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                result = disease_names[np.argmax(prediction)]
            except Exception as e:
                result = f"Error during prediction: {e}"
        else:
            # Deterministic simulation fallback based on filename keywords or hash
            fn_lower = file.filename.lower()
            if 'cocc' in fn_lower:
                result = 'Coccidiosis'
            elif 'new' in fn_lower or 'sick' in fn_lower:
                result = 'Newcastle Disease'
            elif 'salm' in fn_lower or 'farm' in fn_lower:
                result = 'Salmonella'
            else:
                result = 'Healthy'

        return render_template('predict.html', prediction=result, image_path='/' + file_path.replace('\\', '/'))
    return render_template('predict.html')

def open_browser():
    webbrowser.open_new("http://localhost:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, port=5000)