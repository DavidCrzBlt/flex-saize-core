# scripts/run_train.py
import argparse, subprocess, os
from flexsaize.models.train import TrainConfig, RFRegressorTrainer

def short_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-input", default="data/data.csv")
    p.add_argument("--out-clean", default="data/data_clean.csv")
    p.add_argument("--experiment-name", default="FlexSAIze/Banners_RFRegressor")
    p.add_argument("--run-name", default="rf_baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--csv-version", default="unknown")
    p.add_argument("--dvc-rev", default="unknown")
    args = p.parse_args()

    experiment_description = (
        "This is the FlexSAIze regression baseline using RandomForest. "
        "Targets: [x,y,width,height]. Group-aware split, Yeo-Johnson + MinMax."
    )
    experiment_tags = {
        "project_name": "flexsaize",
        "module": "layout-regression",
        "team": "mna-team",
        "project_quarter": "Q4-2025",
        "mlflow.note.content": experiment_description,
    }
    run_tags = {
        "git_commit": short_git_sha(),
        "dvc_rev": args.dvc_rev,
        "csv_version": args.csv_version,
        "model_name": "rf_regressor",
        "seed": str(args.seed),
    }

    cfg = TrainConfig(
        raw_input_path=args.raw_input,
        preprocessed_output_path=args.out_clean,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        experiment_description=experiment_description,
        experiment_tags=experiment_tags,
        run_tags=run_tags,
    )

    trainer = RFRegressorTrainer(cfg)
    metrics = trainer.run()
    print("TEST macro metrics:", metrics)

if __name__ == "__main__":
    main()
