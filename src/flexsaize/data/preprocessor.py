import pandas as pd


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
        if self.df is not None:
            # Si el 'Background' ya fue eliminado o no existe evita que haya errores
            self.df.drop(index='Background',inplace=True,errors='ignore')
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
        # 1. Carga (método de E/S)
        self.load_data()

        # 2. Limpieza 
        if self.df is not None and not self.df.empty:
            self.clean_data()

        # 3. Guardado 
        if self.df is not None and not self.df.empty:
            self.save()
        else:
            # Manejo de error si el DF quedó vacío por limpieza
            print("ERROR: DataFrame vacío. No se guarda el archivo de salida.")

        return self

