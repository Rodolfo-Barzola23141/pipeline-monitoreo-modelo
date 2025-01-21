import google.cloud.aiplatform as aiplatform
from google.cloud import storage
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración inicial
BUCKET_NAME = 'model-monitoring-project-bucket'
PREDICCIONES_URI = 'gs://model-monitoring-project-bucket/predicciones/predicciones.csv'
METRICAS_ENTRENAMIENTO_URI = 'gs://model-monitoring-project-bucket/metricas_entrenamiento/metricas_entrenamiento.json'
METRICAS_URI = 'metricas/metricas_comparacion.json'
COMPARACION_URI = 'comparacion/comparacion_metricas.json'

# Inicializa AI Platform
aiplatform.init(project="project-model-monitoring", location="us-central1")

# Función para cargar datos desde GCS
def load_data_from_gcs(uri):
    """Carga el contenido de un archivo desde GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(uri.replace(f'gs://{BUCKET_NAME}/', ''))
    return blob.download_as_text()

# Cargar y procesar predicciones con delimitador correcto
try:
    predicciones_data = pd.read_csv(PREDICCIONES_URI, sep=';')
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# Asegurarse de que las columnas existen
if 'true_values' not in predicciones_data.columns or 'predictions' not in predicciones_data.columns:
    print("Error: Las columnas 'true_values' y 'predictions' deben estar presentes en el archivo CSV.")
    exit()

# Asignar correctamente las columnas a las variables
ground_truth = predicciones_data['true_values']
predicciones = predicciones_data['predictions']

# Cargar métricas de entrenamiento
try:
    metricas_data = json.loads(load_data_from_gcs(METRICAS_ENTRENAMIENTO_URI))
except Exception as e:
    print(f"Error al cargar las métricas de entrenamiento: {e}")
    exit()

# Función para calcular métricas
def calculate_metrics(y_true, y_pred):
    """Calcula MSE, MAE y R2 entre las predicciones y los valores reales."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# Calcular métricas actuales
mse, mae, r2 = calculate_metrics(ground_truth, predicciones)
metrics = {'mse': mse, 'mae': mae, 'r2': r2}

# Guardar métricas en GCS
try:
    metrics_json = json.dumps(metrics)
    blob = storage.Client().bucket(BUCKET_NAME).blob(METRICAS_URI)
    blob.upload_from_string(metrics_json, content_type='application/json')
    print("Métricas actuales guardadas con éxito.")
except Exception as e:
    print(f"Error al guardar las métricas: {e}")
    exit()

# Comparar métricas actuales con las de entrenamiento
def compare_metrics(training_metrics, current_metrics):
    """Compara las métricas actuales con las de entrenamiento."""
    return {
        'mse_diff': training_metrics['mse'] - current_metrics['mse'],
        'mae_diff': training_metrics['mae'] - current_metrics['mae'],
        'r2_diff': training_metrics['r2'] - current_metrics['r2'],
    }

comparison = compare_metrics(metricas_data, metrics)

# Guardar comparación de métricas
try:
    comparison_json = json.dumps(comparison)
    blob_comparison = storage.Client().bucket(BUCKET_NAME).blob(COMPARACION_URI)
    blob_comparison.upload_from_string(comparison_json, content_type='application/json')
    print("Comparación de métricas guardada con éxito.")
except Exception as e:
    print(f"Error al guardar la comparación de métricas: {e}")
    exit()

print("Pipeline ejecutado con éxito.")
