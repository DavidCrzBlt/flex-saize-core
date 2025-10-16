from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

@dataclass
class SplitConfig:
    test_size_total: float = 0.20  # para temp (val+test)
    val_ratio_within_temp: float = 0.50  # del 20% → 10% val, 10% test
    seed: int = 42

class GroupedSplitter:
    """Realiza split 80/10/10 usando GroupShuffleSplit por 'file' (sin fugas)."""

    def __init__(self, cfg: SplitConfig):
        self.cfg = cfg

    def split(self, X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
        # 1) 80% train vs 20% temp
        gss1 = GroupShuffleSplit(n_splits=1, test_size=self.cfg.test_size_total, random_state=self.cfg.seed)
        train_idx, temp_idx = next(gss1.split(X, y, groups))

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]

        X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
        groups_temp = groups.iloc[temp_idx]

        # 2) Del 20% → mitad val, mitad test
        gss2 = GroupShuffleSplit(n_splits=1, test_size=self.cfg.val_ratio_within_temp, random_state=self.cfg.seed)
        val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))

        X_val,  y_val  = X_temp.iloc[val_idx],  y_temp.iloc[val_idx]
        X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

        groups_val  = groups_temp.iloc[val_idx]
        groups_test = groups_temp.iloc[test_idx]

        # Sanidad básica (opcional): sin intersecciones
        # (dejar prints si te ayudan a validar)
        # print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

        return {
            "train": (X_train, y_train, groups_train),
            "val":   (X_val,   y_val,   groups_val),
            "test":  (X_test,  y_test,  groups_test),
        }
