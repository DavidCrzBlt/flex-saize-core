import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    '''
    En esta clase se van a preparar los datos en crudo. Actualmente solo contiene tres métodos.
    __init__
    load_data
    save
    '''
    def __init__(self, config):
        '''
        Inicialización de la clase
        input_path: str -> Dirección del archivo crudo
        output_path: str -> Dirección del archivo limpio
        df: object -> df vacío
        '''
        self.input_path = config.input_path
        self.output_path = config.output_path
        self.df = None
        self.le = {} # Diccionario de Label Encoder
        self.ptfs = {} # Diccionario de Power Transformer
        self.scaler = {} # Diccionario de MinMax

    def load_data(self):
        '''
        Carga el df desde el input_path
        '''
        try:
            self.df = pd.read_csv(self.input_path)
        except FileNotFoundError:
            print(f'Error: Archivo no encontrado en la ruta: {self.input_path}')
            self.df = pd.DataFrame() # Crea un df vacío para evitar errores posteriores
        return self

    def clean_data(self):
        '''
        Limpieza del df 
        '''
        if self.df is not None and not self.df.empty:
            
            initial_rows = self.df.shape[0]
            # Si el 'Background' ya fue eliminado o no existe evita que haya errores
            self.df.drop(index='BACKGROUND',inplace=True,errors='ignore')
            # Quitamos las filas que son NA
            self.df.dropna(how='any',inplace=True)
            final_rows = self.df.shape[0]
            print(f"Limpieza: Se eliminaron {initial_rows - final_rows} filas.")

        return self
    
    def feature_engineering(self):
        '''
        Crea columnas nuevas a partir de las existentes
        Nuevas variables: relative_area y aspect_ratio (banner y KV)
        '''

        # 1. Verificación obligatoria del DataFrame
        if self.df is None or self.df.empty:

            print("ADVERTENCIA: DataFrame no cargado o está vacío. Saltando Feature Engineering.")
            return self
            
        # Columnas necesarias para el cálculo
        required_cols = ['width', 'height', 'canvas_width', 'canvas_height', 'kv_width', 'kv_height', 'kv_canvas_width', 'kv_canvas_height']

        # 2. Verificar la existencia de todas las columnas requeridas
        if set(required_cols).issubset(self.df.columns):
            
            # 3. Realizar la ingeniería de características
            print("Creando nuevas características (relative_area y aspect_ratio; banner y KV)...")

            # Nuevas variables: relative_area y aspect_ratio (banner y KV)
            # Banner (siempre definidas)

            self.df["aspect_ratio"]  = self.df["width"] / self.df["height"]
            self.df["relative_area"] = (self.df["width"] * self.df["height"]) / (self.df["canvas_width"] * self.df["canvas_height"])

            # KV (solo cuando hay datos)

            self.df["kv_aspect_ratio"] = self.df["kv_width"] / self.df["kv_height"]
            self.df["kv_relative_area"] = (self.df["kv_width"] * self.df["kv_height"]) / (self.df["kv_canvas_width"] * self.df["kv_canvas_height"])
                        
        else:
            # 4. Manejar el caso de columnas faltantes
            missing_cols = set(required_cols) - set(self.df.columns)
            print(f"ADVERTENCIA: No se pudo realizar la Ingeniería de Características. Faltan las siguientes columnas: {missing_cols}")

        return self
            
    def label_encoder(self):
        '''
        Convierte las dos variables categóricas en números cardinales
        '''
        # 1. Verificación obligatoria del DataFrame
        if self.df is None or self.df.empty:

            print("ADVERTENCIA: DataFrame no cargado o está vacío. Saltando Label Encoding.")
            return self
        
        # 2. Definimos las variables a transformar
        variables_le = ['layer_name', 'type']

        # Si las columnas no existen solo transformar las que haya
        cols_to_transform = [col for col in variables_le if col in self.df.columns]

        if not cols_to_transform:
            missing_cols = set(variables_le)-set(self.df.columns)
            print(f"ADVERTENCIA: No se pudo aplicar Label Encoder porque faltan columnas: {missing_cols}")
            return self

        for col in cols_to_transform:
            le = LabelEncoder()

            try:
                self.df[col] = le.fit_transform(self.df[col])
                # Guarda el Label Encoder en el diccionario para los datos en producción
                self.le[col] = le
                print(f"Columna '{col}' codificada y encoder almacenado.")

            except ValueError as e:
                print(f"ADVERTENCIA: No se pudo aplicar Label Encoder porque faltan columnas {cols_to_transform}: {e}")

        return self

    def power_transformer(self):
        '''
        Aplicar transformación Yeo-Johnson para normalizar la frecuencia de los datos
        '''
        # 1. Verificación obligatoria del DataFrame
        if self.df is None or self.df.empty:

            print("ADVERTENCIA: DataFrame no cargado o está vacío. Saltando Power Transformer.")
            return self
        
        # 2. Definiendo variables a transformar
        variables_trans = ['aspect_ratio', 'relative_area', 'kv_aspect_ratio', 'kv_relative_area']

        # Si las columnas no existen solo transformar las que haya
        cols_to_transform = [col for col in variables_trans if col in self.df.columns]

        if not cols_to_transform:
            missing_cols = set(variables_trans)-set(self.df.columns)
            print(f"ADVERTENCIA: No se pudo aplicar Power Transformer porque faltan columnas: {missing_cols}")
            return self
        
        print(f"Aplicando Power Transformer con Yeo-Johnson a las variables: {cols_to_transform}")

        transformer = PowerTransformer(method='yeo-johnson', standardize=False)

        # 3. Aplicar transformaciones
        try:
            transformer.fit(self.df[cols_to_transform])
            self.df[cols_to_transform] = transformer.transform(self.df[cols_to_transform])

            # Guarda el transformer en el diccionario para los datos en producción
            self.ptfs['yeo-johnson-transformer'] = transformer
        except ValueError as e:
            print(f"Falló la transformación Yeo-Johnson en las columnas {cols_to_transform}: {e}")

        return self
        
    def min_max(self):
        '''
        Escalar al rango [0,1] las variables numéricas
        '''
        # 1. Verificación obligatoria del DataFrame
        if self.df is None or self.df.empty:

            print("ADVERTENCIA: DataFrame no cargado o está vacío. Saltando Power Transformer.")
            return self
        
        # 2. Definiendo variables a transformar
        variables_minmax = ['x', 'y', 'width', 'height',
            'kv_x', 'kv_y', 'kv_width', 'kv_height',
            'aspect_ratio', 'relative_area',
            'kv_aspect_ratio', 'kv_relative_area']
        
        # Si las columnas no existen solo transformar las que haya
        cols_to_transform = [col for col in variables_minmax if col in self.df.columns]

        if not cols_to_transform:
            missing_cols = set(variables_minmax)-set(self.df.columns)
            print(f"ADVERTENCIA: No se pudo aplicar MinMax porque faltan columnas: {missing_cols}")
            return self

        print(f"Aplicando MinMax Scaler a las variables: {cols_to_transform}")
            
        scaler = MinMaxScaler()

        # 3. Aplicar transformaciones
        try:
            scaler.fit(self.df[cols_to_transform])
            self.df[cols_to_transform] = scaler.transform(self.df[cols_to_transform])

            # Guarda el scaler en el diccionario para los datos en producción
            self.scaler['MinMax Scaler'] = scaler

        except ValueError as e:
            print(f"Falló el MinMax Scaler en las columnas {cols_to_transform}: {e}")

        return self

    def save(self):
        '''
        Guarda el df limpio en la ruta especificada
        '''
        self.df.to_csv(self.output_path, index=False)
        print(f'Datos guardados en {self.output_path}')
        return self
    
    def run(self):
        '''
        Ejecuta el pipeline completo de carga, limpieza y guardado
        '''
        # Carga (método de E/S)
        self.load_data()

        # Limpieza 
        if self.df is not None and not self.df.empty:
            self.clean_data()

        # Feature Engineering (Creación de nuevas variables) 
        if self.df is not None and not self.df.empty:
            self.feature_engineering()

        # Label Encoder (Cambiar variables categóricas a ordinales) 
        if self.df is not None and not self.df.empty:
            self.label_encoder()

        # Power Transformer (Normalizar la distribución de los datos) 
        if self.df is not None and not self.df.empty:
            self.power_transformer()

        # Min Max Scaler (Escalar las variables numéricas en un rango de 0 a 1) 
        if self.df is not None and not self.df.empty:
            self.min_max()

        # Guardado 
        if self.df is not None and not self.df.empty:
            self.save()
        else:
            # Manejo de error si el DF quedó vacío por limpieza
            print("ERROR: DataFrame vacío. No se guarda el archivo de salida.")

        return self

