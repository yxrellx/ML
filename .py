import onnx
import onnxruntime as ort

# Carga el modelo ONNX
onnx_model_path = 'models/my_modelcov.onnx'
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

# Crea una sesión de inferencia
ort_session = ort.InferenceSession(onnx_model_path)

# Función para realizar predicciones
def predict(image_path):
    import numpy as np
    from PIL import Image

    # Preprocesa la imagen
    image = Image.open(image_path).resize((224, 224))
    img_data = np.array(image).astype('float32')
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data.transpose(0, 3, 1, 2)  # Si tu modelo ONNX espera este formato

    # Realiza la predicción
    inputs = {ort_session.get_inputs()[0].name: img_data}
    outputs = ort_session.run(None, inputs)

    return outputs

# Prueba la función de predicción
result = predict('path/to/your/image.jpg')
print(result)
