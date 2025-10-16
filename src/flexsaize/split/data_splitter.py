import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np
import os

class DataSplitter:
    """
    Clase para realizar la división de datos (Train, Val, Test), 
    garantizando que las filas del mismo grupo permanezcan juntas.
    """
    def __init__(self, config):
        
        self.input_path = config.input_path_clean
        self.output_path_train = config.output_path_train
        self.output_path_val = config.output_path_val
        self.output_path_test = config.output_path_test
    
        self.test_size = config.test_size   
        self.val_size = config.val_size     
        self.group_column = config.group_column 
        
        self.df = None
        self.df_train = None
        self.df_val = None
        self.df_test = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
        except FileNotFoundError:
            print(f"ERROR: Archivo limpio no encontrado en: {self.input_path}")
            self.df = pd.DataFrame()
        return self

    def group_split(self):
        # 1. Verificación obligatoria
        if self.df is None or self.df.empty:
            print("ADVERTENCIA: DataFrame vacío. Saltando Split.")
            return self
        
        # 2. Obtener los nombres únicos de los grupos (ej. file1, file2, ...)
        if self.group_column not in self.df.columns:
            print(f"ERROR: Columna de grupo '{self.group_column}' no encontrada. Usando split aleatorio simple.")
            # Si no hay columna de grupo, caemos a un split simple
            
            return self

        groups = self.df[self.group_column]
        unique_groups = groups.unique()
        
        # 3. Calcular los tamaños de los splits
        
        test_fraction = self.test_size
        
        val_fraction_of_remaining = self.val_size / (1.0 - self.test_size) if (1.0 - self.test_size) > 0 else 0

        # 4. Primera División: Separar TEST del resto
        
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=42)
        
        # Obtenemos los índices para Test y para el resto (Train + Val)
        train_val_idx, test_idx = next(gss_test.split(self.df, groups=groups))
        
        df_train_val = self.df.iloc[train_val_idx]
        self.df_test = self.df.iloc[test_idx]
        
        # 5. Segunda División: Separar VAL de Train
        
        groups_train_val = df_train_val[self.group_column]
        
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_remaining, random_state=42)
        
        # Obtenemos los índices para Val y para Train
        train_idx, val_idx = next(gss_val.split(df_train_val, groups=groups_train_val))
        
        self.df_train = df_train_val.iloc[train_idx]
        self.df_val = df_train_val.iloc[val_idx]

        print(f"Split final: Train={len(self.df_train)}, Val={len(self.df_val)}, Test={len(self.df_test)}")
        return self

    def save_splits(self):
        # 1. Verificación obligatoria
        if self.df_train is None or self.df_train.empty:
             print("ADVERTENCIA: No hay datos para guardar.")
             return self
        
        # Columna a eliminar
        col_to_drop = self.group_column

        print(f"Eliminando la columna de grupo '{col_to_drop}' de los conjuntos Train, Val, y Test.")
             
        # Guardar los archivos de salida
        try:
            # Eliminar la columna antes de guardar
            if col_to_drop in self.df_train.columns:
                self.df_train = self.df_train.drop(columns=[col_to_drop])
            
            if col_to_drop in self.df_val.columns:
                self.df_val = self.df_val.drop(columns=[col_to_drop])
            
            if col_to_drop in self.df_test.columns:
                self.df_test = self.df_test.drop(columns=[col_to_drop])

            # Guardar los archivos de salida
            self.df_train.to_csv(self.output_path_train, index=False)
            self.df_val.to_csv(self.output_path_val, index=False)
            self.df_test.to_csv(self.output_path_test, index=False)
            
            print(f"Archivos de split guardados: Train, Val, Test listos para el entrenamiento.")
            
        except Exception as e:
            print(f"ERROR al guardar los splits o eliminar la columna: {e}")
            
        return self

    def run(self):

        self.load_data()

        if self.df is not None and not self.df.empty:
            self.group_split()

        if self.df is not None and not self.df_train.empty:    
            self.save_splits()
        return self