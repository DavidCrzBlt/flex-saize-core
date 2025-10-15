from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

@dataclass
class EvalResult:
    per_target: Dict[str, Dict[str, float]]  # {target: {"rmse":..., "mae":..., "r2":...}}
    macro_avg: Dict[str, float]              # promedio simple sobre objetivos

class Evaluator:
    """Métricas de regresión multi-salida (por columna y promedio)."""

    @staticmethod
    def compute(y_true: pd.DataFrame, y_pred: np.ndarray, targets: List[str]) -> EvalResult:
        per: Dict[str, Dict[str, float]] = {}
        rmses, maes, r2s = [], [], []

        for i, col in enumerate(targets):
            yt = y_true[col].to_numpy()
            yp = y_pred[:, i] if y_pred.ndim == 2 else y_pred  # por si fuera 1D
            rmse = root_mean_squared_error(yt, yp)
            mae  = mean_absolute_error(yt, yp)
            r2   = r2_score(yt, yp)
            per[col] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
            rmses.append(rmse); maes.append(mae); r2s.append(r2)

        macro = {"rmse": float(np.mean(rmses)), "mae": float(np.mean(maes)), "r2": float(np.mean(r2s))}
        return EvalResult(per_target=per, macro_avg=macro)

    @staticmethod
    def to_mlflow_dict(res: EvalResult) -> Dict[str, float]:
        out = {}
        # macro
        out["rmse_macro"] = res.macro_avg["rmse"]
        out["mae_macro"]  = res.macro_avg["mae"]
        out["r2_macro"]   = res.macro_avg["r2"]
        # por target
        for t, m in res.per_target.items():
            out[f"rmse_{t}"] = m["rmse"]
            out[f"mae_{t}"]  = m["mae"]
            out[f"r2_{t}"]   = m["r2"]
        return out
