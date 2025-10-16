import argparse
from flexsaize.data.preprocessor import DataPreprocessor, PreprocessConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = PreprocessConfig(
        input_path=args.input,
        output_path=args.output
    )

    pre = DataPreprocessor(cfg)
    df_tr, groups = pre.run()

if __name__ == "__main__":
    main()
    