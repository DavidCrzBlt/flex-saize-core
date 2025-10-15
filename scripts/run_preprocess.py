from flexsaize.data.preprocessor import DataPreprocessor

def main():
    input_path = "data/data.csv"
    output_path = "data/data_clean.csv"

    pre = DataPreprocessor(input_path,output_path)
    pre.load_data()
    pre.remove_empty_rows()
    pre.save_cleaned()

if __name__ == "__main__":
    main()
    