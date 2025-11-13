import argparse
import mlflow
import os
from dotenv import load_dotenv
from flexsaize.data.preprocessor import DataPreprocessor

def main():

    # Cargamos la variable de entorno
    load_dotenv()
    
    required_envs = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY"
    ]

    for var in required_envs:
        value = os.getenv(var)
        if not value:
            print(f"⚠️ Advertencia: variable {var} no encontrada en .env")
        else:
            os.environ[var] = value

    print("\n--- DEBUG DE MLFLOW ---")
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        print(f"✅ Variable de entorno encontrada: MLFLOW_TRACKING_URI = {tracking_uri}")
    else:
        print("❌ ERROR: ¡No se encontró la variable MLFLOW_TRACKING_URI en tu .env!")
        print("    Asegúrate que el archivo .env esté en la raíz del proyecto (donde corres 'dvc').")
        return # Salir si no hay URI

    try:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ MLflow URI configurada en: {mlflow.get_tracking_uri()}")
        print(f"📦 Artifact store: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")

    except Exception as e:
        print(f"❌ ERROR al configurar la URI de MLflow: {e}")
        return
        
    print("--- FIN DEBUG ---")
    # --- FIN DEL BLOQUE DE DEBUG ---
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

    print("📦 Artifact store endpoint:", os.getenv("MLFLOW_S3_ENDPOINT_URL"))

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

        # Registrar la métrica final si se calculó
        if preprocessor.df is not None and not preprocessor.df.empty:
            mlflow.log_metric("final_dataset_rows", preprocessor.df.shape[0])
        else:
            print("No se registraron métricas, el DataFrame está vacío.")

        print(f"Subiendo artefactos desde {args.transformers_path}...")
        try:
            print(f"Subiendo artefactos (receta) desde {args.transformers_path}...")
            mlflow.log_artifacts(args.transformers_path, artifact_path="preprocessor_recipe")
            print("✅ ¡Artefactos logueados exitosamente!")
            
        except Exception as e:
            # Si falla, esto nos dirá por qué (ej. "Conexión rechazada")
            print(f"❌ ERROR AL SUBIR ARTEFACTOS: {e}")
            print("    Revisa la conexión de red con la RPi y los permisos del artifact store.")
        
        print("¡Run de preprocesamiento completado y artefactos logueados!")

if __name__ == "__main__":
    main()
