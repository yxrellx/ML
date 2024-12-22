import onnxruntime as ort
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Cargar el modelo ONNX
model_path = 'models/my_modelcov.onnx'
ort_session = ort.InferenceSession(model_path)

# Definir las clases del modelo
classes = ["batman", "ironman", "linternaverde", "spiderman", "wolwerine"]

def preprocess_image(image):
    # Redimensionar la imagen a 224x224 píxeles
    image = image.resize((224, 224))
    # Convertir la imagen a un array numpy
    img_data = np.array(image).astype('float32')
    # Normalizar los valores de píxeles si es necesario (ajustar según sea necesario)
    img_data = img_data / 255.0
    # Añadir una dimensión para el batch
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file)
        img_data = preprocess_image(image)

        inputs = {ort_session.get_inputs()[0].name: img_data}
        outputs = ort_session.run(None, inputs)

        # Obtener la predicción
        predictions = outputs[0][0]

        # Mapear las predicciones a las clases
        predicted_class = classes[np.argmax(predictions)]
        predicted_probabilities = {classes[i]: float(predictions[i]) for i in range(len(classes))}

        return jsonify({'predicted_class': predicted_class, 'predicted_probabilities': predicted_probabilities})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
