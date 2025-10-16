import argparse
import mlflow
import os
from dotenv import load_dotenv
from flexsaize.data.preprocessor import DataPreprocessor

def main():

    # Cargamos la variable de entorno
    load_dotenv()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # Agregamos los parámetros de nuestra clase
    parser = argparse.ArgumentParser(description="Preprocesamiento de  datos")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Ruta al archivo de datos crudos")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Ruta donde se guardara el archivo limpio")
    
    # Aquí creamos el objeto args
    args = parser.parse_args()

    # FUERZA LA URI DE SEGUIMIENTO DENTRO DEL SCRIPT
    mlflow.set_tracking_uri(tracking_uri)

    # Si el experimento no existe, MLflow lo crea automáticamente.
    mlflow.set_experiment("Flexsaize_Preprocessing_Pipeline")

    # Esto envuelve todo el trabajo de preprocesamiento en un registro de MLflow
    with mlflow.start_run(run_name="Data_Preprocess_Stage") as run:

        # Registrar la configuración usada
        mlflow.log_params(vars(args))

        # El objeto 'args' (que contiene input_path y output_path) es nuestro 'config'
        preprocessor = DataPreprocessor(config=args)
        # Ejecutar el pipeline completo
        preprocessor.run()

        # Registrar la métrica final si se calculó
        mlflow.log_metric("final_dataset_rows", preprocessor.df.shape[0])

if __name__ == "__main__":
    main()
    