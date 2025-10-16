import pandas as pd


class DataPreprocessor:
    '''
    En esta clase se van a preparar los datos en crudo. Actualmente solo contiene tres métodos.
    __init__
    load_data
    save
    '''
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        return self

    def clean_data(self):
        self.df.drop(index='Background',inplace=True)
        return self

    def save(self):
        return self.df.to_csv(self.output_path, index=False)
