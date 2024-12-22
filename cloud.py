from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Sube un archivo a un bucket de Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"Archivo {source_file_name} subido a {destination_blob_name}.")

# Ejemplo de uso:
upload_blob('your-bucket-name', 'models/my_modelcov.onnx', 'my_modelcov.onnx')
