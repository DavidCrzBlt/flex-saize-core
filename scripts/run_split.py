import argparse
import mlflow
import os
from dotenv import load_dotenv
from flexsaize.split.data_splitter import DataSplitter

def main():

    # Cargamos la variable de entorno
    load_dotenv()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # Agregamos los parámetros de nuestra clase
    parser = argparse.ArgumentParser(description="Separación de los datos")
    parser.add_argument("--input-path-clean", type=str, required=True,
                        help="Ruta al archivo de datos limpios")
    parser.add_argument("--output-path-train", type=str, required=True,
                        help="Ruta donde se guardarán los datos de 'Train'")
    parser.add_argument("--output-path-val", type=str, required=True,
                        help="Ruta donde se guardarán los datos de 'Validación'")
    parser.add_argument("--output-path-test", type=str, required=True,
                        help="Ruta donde se guardarán los datos de 'Test'")
    parser.add_argument("--test-size", type=float, required=True,
                        help="Porcentaje de (0,1) de tamaño de 'Test'")
    parser.add_argument("--val-size", type=float, required=True,
                        help="Porcentaje de (0,1) de tamaño de 'Val'")
    parser.add_argument("--group-column", type=str, required=True,
                        help="Nombre de la columna de grupos")
    
    # Aquí creamos el objeto args
    args = parser.parse_args()

    # FUERZA LA URI DE SEGUIMIENTO DENTRO DEL SCRIPT
    mlflow.set_tracking_uri(tracking_uri)

    # Si el experimento no existe, MLflow lo crea automáticamente.
    mlflow.set_experiment("Flexsaize_Split_Pipeline")

    # Esto envuelve todo el trabajo de preprocesamiento en un registro de MLflow
    with mlflow.start_run(run_name="Data_Split_Stage") as run:

        # Registrar la configuración usada
        mlflow.log_params(vars(args))

        # El objeto 'args' (que contiene input_path y output_path) es nuestro 'config'
        data_splitter = DataSplitter(config=args)
        # Ejecutar el pipeline completo
        data_splitter.run()

        # Verificamos que los DataFrames de salida existan y no estén vacíos.
        if data_splitter.df_train is not None and not data_splitter.df_train.empty:
            mlflow.log_metric("train_rows", len(data_splitter.df_train))
            mlflow.log_metric("validation_rows", len(data_splitter.df_val))
            mlflow.log_metric("test_rows", len(data_splitter.df_test))
        else:
            print("ADVERTENCIA: No se pudo registrar el tamaño de los splits (los DataFrames están vacíos).")
            mlflow.log_metric("total_rows", 0)

if __name__ == "__main__":
    main()
    