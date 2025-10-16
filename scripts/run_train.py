import argparse
import json
import mlflow
import os
from dotenv import load_dotenv
from flexsaize.models.model_trainer import ModelTrainer

def main():

    # Cargamos la variable de entorno
    load_dotenv()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # Agregamos los parámetros de nuestra clase
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos")
    parser.add_argument("--input-path-train", type=str, required=True,
                        help="Ruta a archivo de 'Train'")
    parser.add_argument("--input-path-val", type=str, required=True,
                        help="Ruta a archivo de 'Validación'")
    parser.add_argument("--input-path-test", type=str, required=True,
                        help="Ruta a archivo de 'Test'")
    parser.add_argument("--target-col", type=str, required=True,
                        help="Columnas objetivo que el modelo va a predecir")
    parser.add_argument("--model-type", type=str, required=True,
                        help="Nombre del modelo que se quiere entrenar")
    parser.add_argument("--hyperparams", type=str, required=True,
                        help="Diccionario de hiperparámetros del modelo")
    
    # Aquí creamos el objeto args
    args = parser.parse_args()
    args.target_col = args.target_col.split(',')

    # Intentamos cargar la cadena JSON como un diccionario
    try:
        args.hyperparams = json.loads(args.hyperparams) # Transforma la cadena JSON en dict
    except json.JSONDecodeError as e:
        print(f"ERROR: El argumento --hyperparams no es un JSON válido. Detalle: {e}")

    # FUERZA LA URI DE SEGUIMIENTO DENTRO DEL SCRIPT
    mlflow.set_tracking_uri(tracking_uri)

    # Si el experimento no existe, MLflow lo crea automáticamente.
    mlflow.set_experiment("Flexsaize_Training_Models_Pipeline")

    # Esto envuelve todo el trabajo de preprocesamiento en un registro de MLflow
    with mlflow.start_run(run_name="Models_Training_Stage") as run:

        # Registrar la configuración usada
        mlflow.log_params(vars(args))

        # El objeto 'args' (que contiene input_path y output_path) es nuestro 'config'
        preprocessor = ModelTrainer(config=args)
        # Ejecutar el pipeline completo
        preprocessor.run()

if __name__ == "__main__":
    main()
    