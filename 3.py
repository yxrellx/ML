from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image

app = Flask(__name__)

# Cargar el modelo ONNX
model_path = 'models/my_modelcov.onnx'
ort_session = ort.InferenceSession(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).resize((224, 224))
    img_data = np.array(image).astype('float32')
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data.transpose(0, 3, 1, 2)  # Si tu modelo ONNX espera este formato

    inputs = {ort_session.get_inputs()[0].name: img_data}
    outputs = ort_session.run(None, inputs)

    return jsonify({'prediction': outputs[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)
