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
    parser.add_argument("--output-path-transformers", type=str, required=True,
                        dest="transformers_path",
                        help="Ruta donde se guardaran los transformers limpios")
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

        mlflow.log_artifacts(args.transformers_path, artifact_path="preprocessor_recipe")

        # Registrar la métrica final si se calculó
        if preprocessor.df is not None and not preprocessor.df.empty:
            mlflow.log_metric("final_dataset_rows", preprocessor.df.shape[0])
        else:
            print("No se registraron métricas, el DataFrame está vacío.")

        print(f"Subiendo artefactos desde {args.transformers_path}...")
        mlflow.log_artifacts(args.transformers_path, artifact_path="preprocessor_recipe")
        
        print("¡Run de preprocesamiento completado y artefactos logueados!")

if __name__ == "__main__":
    main()
