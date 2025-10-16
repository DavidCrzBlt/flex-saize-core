import pandas as pd
import numpy as np
import mlflow

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from flexsaize.models.model_config import MODEL_MAPPER, MODEL_CLASSES

class ModelTrainer:
    def __init__(self, config):
        '''
        Inicialización de la clase
        '''
        self.input_path_train = config.input_path_train
        self.input_path_val = config.input_path_val
        self.input_path_test = config.input_path_test
        self.target_col = config.target_col
        self.model_type = config.model_type
        self.hyperparams = config.hyperparams

        self.model = None # Almacena el modelo entrenado

        self.df_train = None
        self.X_train = None 
        self.y_train = None

        self.df_val = None
        self.X_val = None 
        self.y_val = None
        
        self.df_test = None
        self.X_test = None 
        self.y_test = None

        self.y_pred_val = None
        self.y_pred_test = None
        self.metrics = {}


    def load_data(self):
        '''
        Cargamos los df train, val y test desde los input paths
        '''
        # Intentamos leer el df para train            
        try:
            self.df_train = pd.read_csv(self.input_path_train)

            for col in self.df_train.select_dtypes(include=['int64']).columns:
                # Solo conviértelas si no son la columna objetivo
                if col not in self.target_col:
                    self.df_train[col] = self.df_train[col].astype('Int64')

            self.X_train = self.df_train.drop(columns=self.target_col)
            self.y_train = self.df_train[self.target_col].copy()
        except FileNotFoundError:
            print(f'Error: Archivo "train.csv" no encontrado en la ruta: {self.input_path_train}')
            # Inicializa DataFrames vacíos para evitar AttributeError
            self.df_train = pd.DataFrame() 
            self.X_train = pd.DataFrame()
            self.y_train = pd.DataFrame()

        except KeyError as e:
            print(f'Error: Columna objetivo faltante en el archivo de entrenamiento: {e}')
            # Inicializa DataFrames vacíos si falta la columna objetivo
            self.df_train = pd.DataFrame()
            self.X_train = pd.DataFrame()
            self.y_train = pd.DataFrame()

        # Intentamos leer el df para val
        try:
            self.df_val = pd.read_csv(self.input_path_val)

            for col in self.df_val.select_dtypes(include=['int64']).columns:
                # Solo conviértelas si no son la columna objetivo
                if col not in self.target_col:
                    self.df_val[col] = self.df_val[col].astype('Int64')

            self.X_val = self.df_val.drop(columns=self.target_col)
            self.y_val = self.df_val[self.target_col].copy()
        except FileNotFoundError:
            print(f'Error: Archivo "val.csv" no encontrado en la ruta: {self.input_path_val}')
            # Inicializa DataFrames vacíos para evitar AttributeError
            self.df_val = pd.DataFrame() 
            self.X_val = pd.DataFrame()
            self.y_val = pd.DataFrame()

        except KeyError as e:
            print(f'Error: Columna objetivo faltante en el archivo de entrenamiento: {e}')
            # Inicializa DataFrames vacíos si falta la columna objetivo
            self.df_val = pd.DataFrame()
            self.X_val = pd.DataFrame()
            self.y_val = pd.DataFrame()

        # Intentamos leer el df para test            
        try:
            self.df_test = pd.read_csv(self.input_path_test)

            for col in self.df_test.select_dtypes(include=['int64']).columns:
                # Solo conviértelas si no son la columna objetivo
                if col not in self.target_col:
                    self.df_test[col] = self.df_test[col].astype('Int64')

            self.X_test = self.df_test.drop(columns=self.target_col)
            self.y_test = self.df_test[self.target_col].copy()
        except FileNotFoundError:
            print(f'Error: Archivo "test.csv" no encontrado en la ruta: {self.input_path_test}')
            # Inicializa DataFrames vacíos para evitar AttributeError
            self.df_test = pd.DataFrame() 
            self.X_test = pd.DataFrame()
            self.y_test = pd.DataFrame()

        except KeyError as e:
            print(f'Error: Columna objetivo faltante en el archivo de entrenamiento: {e}')
            # Inicializa DataFrames vacíos si falta la columna objetivo
            self.df_test = pd.DataFrame()
            self.X_test = pd.DataFrame()
            self.y_test = pd.DataFrame()

        return self

    def train_model(self):
        '''
        Entrenamiento de modelos del diccionario MODEL_MAPPER
        '''
        # Buscar que el modelo exista en nuestro diccionario
        if self.model_type not in MODEL_MAPPER:
            print(f"El modelo {self.model_type} no está soportado.")
            return self
        
        # Inicializar el modelo
        ModelClass = MODEL_MAPPER[self.model_type]

        try:
            if ModelClass is MultiOutputRegressor:
                # 1. Creamos una instancia del modelo base (ej. XGBRegressor)
                BaseModel = MODEL_CLASSES[self.model_type]
                base_model_instance = BaseModel(**self.hyperparams) 
                
                # 2. Inicializamos el envoltorio usando la instancia del modelo base
                # De esta forma los hyperparams se pasan al modelo base, no al MultiOutputRegressor
                self.model = ModelClass(base_model_instance) 
                
            else:
                # Para modelos nativos (ej. RandomForest)
                self.model = ModelClass(**self.hyperparams)

        except TypeError as e:
            print(f"ERROR: Hiperparámetros inválidos para el modelo {self.model_type}. Detalles: {e}")
            self.model = None
            return self
        
        # Ejecutar el modelo
        if self.X_train.empty or self.y_train.empty:
            print(f"ADVERTENCIA: Los datos de entrenamiento X/y están vacíos. Saltando entrenamiento")
            return self
        
        print(f"Entrenando el modelo {self.model_type} ...")
        self.model.fit(self.X_train,self.y_train)
        print("Entrenamiento completado.")

        return self


    def evaluate_model(self):
        '''
        Evaluación del modelo
        '''
        # Verificamos que haya algo en el modelo
        if self.model is None:
            print(f"ADVERTENCIA: El modelo {self.model_type} está vacío.")
            self.model = None
            return self
        
        # Probamos las predicciones con val
        try:
            print("Probando las predicciones en validación ...")
            self.y_pred_val = self.model.predict(self.X_val)
            rmse_val = root_mean_squared_error(self.y_val, self.y_pred_val, multioutput="raw_values")
            mae_val  = mean_absolute_error(self.y_val, self.y_pred_val, multioutput="raw_values")
            r2_val   = r2_score(self.y_val, self.y_pred_val, multioutput="raw_values")
        except ValueError as e:
            print(f"ERROR: {e}")
            return self
        
        # Probando las predicciones en test
        try:
            print("Probando las predicciones en test ...")
            self.y_pred_test = self.model.predict(self.X_test)
            rmse_test = root_mean_squared_error(self.y_test, self.y_pred_test,multioutput="raw_values")
            mae_test  = mean_absolute_error(self.y_test, self.y_pred_test,multioutput="raw_values")
            r2_test   = r2_score(self.y_test, self.y_pred_test,multioutput="raw_values")
        except ValueError as e:
            print(f"ERROR: {e}")
            return self
        
        # 2. Obtener los nombres de las columnas objetivo
        target_names = self.target_col 
        
        # 3. Función auxiliar para procesar y registrar las métricas
        def log_multioutput_metrics(metrics_array, metric_name, set_name):
            """Asigna cada valor de métrica a una clave única."""
            for i, value in enumerate(metrics_array):
                col_name = target_names[i]
                # Ejemplo de clave: 'val_rmse_x' o 'test_r2_width'
                key = f"{set_name}_{metric_name}_{col_name}"
                self.metrics[key] = value
                
            # Opcional: Registrar el promedio para tener una métrica resumen
            self.metrics[f"{set_name}_{metric_name}_mean"] = np.mean(metrics_array)
        
        # 4. Procesar Métricas de Validación
        log_multioutput_metrics(rmse_val, "rmse", "val")
        log_multioutput_metrics(mae_val, "mae", "val")
        log_multioutput_metrics(r2_val, "r2", "val")
        
        # 5. Procesar Métricas de Prueba (Test)
        log_multioutput_metrics(rmse_test, "rmse", "test")
        log_multioutput_metrics(mae_test, "mae", "test")
        log_multioutput_metrics(r2_test, "r2", "test")
        
        print(f"Métricas calculadas y almacenadas en self.metrics: {len(self.metrics)} valores.")
        print(self.metrics)
        
        return self

    def log_artifacts(self):
        '''
        Registra el modelo, las métricas y los parámetros en MLflow.
        '''
        # 1. Verificar si hay resultados para registrar
        if not self.metrics:
            print("ADVERTENCIA: No hay métricas para registrar. Saltando log de métricas y modelo.")
            return self

        # 2. Registrar las métricas (del diccionario self.metrics)
        print(f"Registrando {len(self.metrics)} métricas en MLflow...")
        for key, value in self.metrics.items():
            mlflow.log_metric(key, value)
            
        # 3. Registrar los hiperparámetros (si es que aún no se registraron en run_train.py)
        mlflow.log_params(self.hyperparams)

        # 4. Registrar el Modelo (Artefacto)
        if self.model is not None:
            print("Registrando el modelo entrenado como artefacto...")

            input_example = self.X_train.head(1)
            
            mlflow.sklearn.log_model(
                sk_model=self.model,
                name="model",  # Carpeta donde se guardará el modelo
                input_example = input_example,
                registered_model_name=f"{self.model_type}_Model" # Nombre en el Model Registry (Opcional pero recomendable)
            )
            print("Modelo registrado con éxito.")

        return self

    def run(self):
        '''
        Ejecuta el pipeline completo de carga, entrenamiento, evaluación y registro.
        '''
        print("Iniciando pipeline de entrenamiento ...")
        
        # 1. Cargar y Separar Datos (X/y)
        self.load_data()
        
        # 2. Entrenar el Modelo
        self.train_model()
        
        # 3. Evaluar el Modelo
        if self.model is not None:
            self.evaluate_model()
        else:
            print("ADVERTENCIA: No se pudo entrenar el modelo. Saltando evaluación.")
            
        # 4. Registrar Artefactos (MLflow)
        self.log_artifacts()
        
        print("Pipeline de entrenamiento completado")
        return self