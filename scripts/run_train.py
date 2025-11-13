import argparse
import yaml
import mlflow
import os
import datetime
from dotenv import load_dotenv
from flexsaize.train.model_trainer import ModelTrainer

def get_active_hyperparams(model_type, params_file='params.yaml'):
    """
    Lee params.yaml y extrae el diccionario de hyperparams del modelo activo.
    """
    
    # 1. Leer el archivo YAML
    try:
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: El archivo de parámetros '{params_file}' no fue encontrado.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error: El archivo de parámetros '{params_file}' tiene un formato YAML inválido. Detalle: {e}")
    
    # 2. Intentar extraer los hyperparams del modelo específico
    try:
        # Navegamos a la sección 'active_hyperparams' y luego al 'model_type'
        hyperparams_dict = params['train']['active_hyperparams'][model_type]
        return hyperparams_dict
    except KeyError as e:
        # Se lanza si 'train', 'active_hyperparams', o 'model_type' no existen
        raise ValueError(f"Error: El modelo '{model_type}' o la estructura 'train.active_hyperparams' no está definida en '{params_file}'.")

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
    
    # Aquí creamos el objeto args
    args = parser.parse_args()
    args.target_col = args.target_col.split(',')

    try:
        # Obtener los hiperparámetros correctos del params.yaml
        args.hyperparams = get_active_hyperparams(args.model_type)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR FATAL DE CONFIGURACIÓN: {e}")
        # Detenemos la ejecución si no se pudo cargar la configuración de hyperparams
        return # Salimos de la función main()

    # FUERZA LA URI DE SEGUIMIENTO DENTRO DEL SCRIPT
    mlflow.set_tracking_uri(tracking_uri)

    # Si el experimento no existe, MLflow lo crea automáticamente.
    mlflow.set_experiment("Flexsaize_Training_Models_Pipeline")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model_type}__{timestamp}"

    # Esto envuelve todo el trabajo de preprocesamiento en un registro de MLflow
    with mlflow.start_run(run_name=run_name) as run:

        dvc_tags = {
            "git_commit": os.environ.get("DVC_EXP_GIT_COMMIT"),
            "dvc_version": os.environ.get("DVC_EXP_NAME"),
            "dvc_baseline_rev": os.environ.get("DVC_BASELINE_REV"),
        }
        # Filtramos valores None y registramos como tags en MLflow
        dvc_tags = {k: v for k, v in dvc_tags.items() if v}
        mlflow.set_tags(dvc_tags)

        # Registrar la configuración usada
        mlflow.log_params(args.hyperparams)

        trainer = ModelTrainer(config=args)

        # Ejecutar el pipeline completo
        trainer.run()

        # -----------------------------------------------------------------
        # 💾 Guardar el modelo entrenado localmente para versionar con DVC
        # -----------------------------------------------------------------
        import joblib
        os.makedirs("models", exist_ok=True)

        model_path = f"models/{args.model_type}_{timestamp}.pkl"
        if hasattr(trainer, "model") and trainer.model is not None:
            joblib.dump(trainer.model, model_path)
            print(f"✅ Modelo guardado en: {model_path}")

            # Registrar también en MLflow
            mlflow.log_artifact(model_path, artifact_path="model")
        else:
            print("⚠️ No se encontró modelo entrenado para guardar.")

if __name__ == "__main__":
    main()
    