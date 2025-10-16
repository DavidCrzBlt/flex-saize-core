import argparse
import mlflow
from flexsaize.data.preprocessor import DataPreprocessor

def main():

    # Agregamos los parámetros de nuestra clase
    parser = argparse.ArgumentParser(description="Preprocesamiento de  datos")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Ruta al archivo de datos crudos")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Ruta donde se guardara el archivo limpio")
    
    # Aquí creamos el objeto args
    args = parser.parse_args()

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
    