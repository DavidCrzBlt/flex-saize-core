from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlflow.tracking import MlflowClient

from flexsaize.data.preprocessor import DataPreprocessor, PreprocessConfig
from flexsaize.data.splitter import GroupedSplitter, SplitConfig
from flexsaize.models.evaluate import Evaluator

from mlflow import sklearn as mlflow_sklearn  

@dataclass
class TrainConfig:
    # paths
    raw_input_path: str              # data/data.csv (o importado por DVC)
    preprocessed_output_path: str    # data/data_clean.csv
    # columnas
    y_cols: Tuple[str, ...] = ("x","y","width","height")
    group_col: str = "file"
    # modelo
    n_estimators: int = 400
    max_depth: Optional[int] = None
    seed: int = 42
    # split
    test_size_total: float = 0.20
    val_ratio_within_temp: float = 0.50
    # MLflow
    experiment_name: str = "FlexSAIze/Banners_RFRegressor"
    run_name: str = "rf_baseline"
    experiment_description: str = (
        "RandomForestRegressor para predicción de [x,y,width,height] en banners. "
        "Preprocesamiento con features (ratios/áreas), encoding y Yeo-Johnson+MinMax."
    )
    experiment_tags: Optional[Dict[str, str]] = None
    run_tags: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.experiment_tags is None:
            self.experiment_tags = {
                "project_name": "flexsaize",
                "module": "layout-regression",
                "team": "mna-team",
                "mlflow.note.content": self.experiment_description,
            }

class RFRegressorTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.client = MlflowClient()
        self.experiment_id = self._get_or_create_experiment()

    def _get_or_create_experiment(self) -> str:
        exp = self.client.get_experiment_by_name(self.cfg.experiment_name)
        if exp is None:
            return self.client.create_experiment(
                name=self.cfg.experiment_name,
                tags=self.cfg.experiment_tags
            )
        # Refrescar nota/tags (opcional)
        try:
            self.client.set_experiment_tag(exp.experiment_id, "mlflow.note.content", self.cfg.experiment_description)
            for k, v in self.cfg.experiment_tags.items():
                self.client.set_experiment_tag(exp.experiment_id, k, v)
        except Exception:
            pass
        return exp.experiment_id

    # ---- PIPELINE COMPLETO ----
    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        pcfg = PreprocessConfig(
            input_path=self.cfg.raw_input_path,
            output_path=self.cfg.preprocessed_output_path,
            y_cols=self.cfg.y_cols,
            group_col=self.cfg.group_col,
        )
        pre = DataPreprocessor(pcfg)
        df_tr, groups = pre.run()

        # Separar X/y
        y = df_tr[list(self.cfg.y_cols)].copy()
        X = df_tr.drop(columns=list(self.cfg.y_cols)).copy()
        return X, y, groups

    def split(self, X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series):
        scfg = SplitConfig(
            test_size_total=self.cfg.test_size_total,
            val_ratio_within_temp=self.cfg.val_ratio_within_temp,
            seed=self.cfg.seed
        )
        splitter = GroupedSplitter(scfg)
        return splitter.split(X, y, groups)

    def build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            random_state=self.cfg.seed,
            n_jobs=-1
        )

    def run(self) -> Dict[str, float]:
        # 1) Preprocess + split
        X, y, groups = self.preprocess()
        splits = self.split(X, y, groups)
        X_tr, y_tr, _ = splits["train"]
        X_va, y_va, _ = splits["val"]
        X_te, y_te, _ = splits["test"]

        # 2) Modelo
        model = self.build_model()
        model.fit(X_tr, y_tr)

        # 3) Eval val + test
        y_pred_val = model.predict(X_va)
        res_val = Evaluator.compute(y_va, y_pred_val, targets=list(self.cfg.y_cols))

        y_pred_test = model.predict(X_te)
        res_test = Evaluator.compute(y_te, y_pred_test, targets=list(self.cfg.y_cols))

        # 4) MLflow logging
        import mlflow
        mlflow.set_experiment(self.cfg.experiment_name)
        with mlflow.start_run(run_name=self.cfg.run_name, experiment_id=self.experiment_id):
            if self.cfg.run_tags:
                mlflow.set_tags(self.cfg.run_tags)

            # Params del modelo y datos
            mlflow.log_params({
                "model": "RandomForestRegressor",
                "n_estimators": self.cfg.n_estimators,
                "max_depth": self.cfg.max_depth,
                "seed": self.cfg.seed,
                "n_features": X.shape[1],
                "targets": ",".join(self.cfg.y_cols),
                "split_train": len(X_tr),
                "split_val":   len(X_va),
                "split_test":  len(X_te),
            })

            # Métricas (val y test) – macro + por target
            from flexsaize.models.evaluate import Evaluator as E
            val_dict  = E.to_mlflow_dict(res_val)
            test_dict = {f"test_{k}": v for k, v in E.to_mlflow_dict(res_test).items()}
            mlflow.log_metrics(val_dict)
            mlflow.log_metrics(test_dict)

            # Modelo
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Devuelve macro de test para consola
        return {"rmse": res_test.macro_avg["rmse"], "mae": res_test.macro_avg["mae"], "r2": res_test.macro_avg["r2"]}
