import pandas as pd

class DataPreprocessor:
    """
    Clase base para limpiar y preparar datasets de FlexSAIze.
    """

    def __init__(self, input_path, output_path=None):
        self.input_path = input_path
        self.output_path = output_path or input_path.replace(".csv", "_clean.csv")
        self.df = None

    def load_data(self):
        """Carga el CSV en un DataFrame de pandas."""
        self.df = pd.read_csv(self.input_path)
        return self.df

    def remove_empty_rows(self):
        """Elimina filas vacías o completamente nulas."""
        if self.df is None:
            raise ValueError("Primero debes cargar el dataset con load_data().")
        self.df.dropna(how='all', inplace=True)
        return self.df

    def save_cleaned(self):
        """Guarda el CSV limpio en la ruta de salida."""
        self.df.to_csv(self.output_path, index=False)
        print(f"Archivo limpio guardado en: {self.output_path}")
