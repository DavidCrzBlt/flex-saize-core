from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PowerTransformer, MinMaxScaler

@dataclass
class PreprocessConfig:
    input_path: str
    output_path: str
    # columnas objetivo
    y_cols: Tuple[str, ...] = ("x", "y", "width", "height")
    # categóricas a codificar
    cat_cols: Tuple[str, ...] = ("layer_name", "type")
    # excluir de Yeo-Johnson/MinMax
    exclude_transform: Tuple[str, ...] = (
        "canvas_width","canvas_height","kv_canvas_width","kv_canvas_height",
        "priority","z_index","kv_z_index"
    )
    # nombre de columna de grupos
    group_col: str = "file"

class DataPreprocessor:
    """
    Aplica el pipeline de features + limpieza que definiste:
    - nuevas variables: aspect_ratio, relative_area (banner y KV)
    - NaN en KV cuando falte algo o is_in_kv == 0
    - eliminar 'Background'
    - label encoding de ('layer_name', 'type')
    - Yeo-Johnson + MinMax [0,1] a numéricas aplicables
    - separa y_cols y conserva 'file' como grupos (aparte)
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.df_tr: Optional[pd.DataFrame] = None
        self.groups: Optional[pd.Series] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.ptfs: Dict[str, PowerTransformer] = {}

    # ---------- CARGA ----------
    def load(self) -> "DataPreprocessor":
        self.df = pd.read_csv(self.cfg.input_path)
        return self

    # ---------- FEATURE ENGINEERING ----------
    def engineer_features(self) -> "DataPreprocessor":
        assert self.df is not None
        df = self.df.copy()

        # Banner
        df["aspect_ratio"]  = df["width"] / df["height"]
        df["relative_area"] = (df["width"] * df["height"]) / (df["canvas_width"] * df["canvas_height"])

        # KV
        df["kv_aspect_ratio"]  = df["kv_width"] / df["kv_height"]
        df["kv_relative_area"] = (df["kv_width"] * df["kv_height"]) / (df["kv_canvas_width"] * df["kv_canvas_height"])

        # Invalidar KV si no aplica
        kv_base = ["kv_x","kv_y","kv_width","kv_height","kv_canvas_width","kv_canvas_height","kv_z_index"]
        kv_derived = ["kv_aspect_ratio","kv_relative_area"]
        kv_all = kv_base + kv_derived

        mask_bad_kv = (df["is_in_kv"] == 0) | df[kv_base].isna().any(axis=1)
        df.loc[mask_bad_kv, kv_all] = np.nan

        # Fuera 'Background'
        df = df[df["layer_name"] != "Background"].copy()

        self.df_tr = df
        return self

    # ---------- ENCODING CATEGÓRICAS ----------
    def encode_categoricals(self) -> "DataPreprocessor":
        assert self.df_tr is not None
        df = self.df_tr

        # Guarda grupos y elimina 'file'
        self.groups = df[self.cfg.group_col].copy()
        df.drop(columns=[self.cfg.group_col], inplace=True, errors="ignore")

        for col in self.cfg.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.df_tr = df
        return self

    # ---------- TRANSFORMACIÓN NUMÉRICA ----------
    def transform_numericals(self) -> "DataPreprocessor":
        assert self.df_tr is not None
        df = self.df_tr

        cat_set = set(self.cfg.cat_cols)
        excl_set = set(self.cfg.exclude_transform)
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        cols_to_tf = [c for c in num_cols if (c not in cat_set) and (c not in excl_set)]

        for col in cols_to_tf:
            s = df[col]
            mask = s.notna()
            if mask.sum() >= 2:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                vals_pt = pt.fit_transform(s[mask].to_numpy().reshape(-1,1))

                mm = MinMaxScaler()
                vals_scaled = mm.fit_transform(vals_pt).ravel()

                df.loc[mask, col] = vals_scaled
                self.ptfs[col] = pt
                self.scalers[col] = mm

        self.df_tr = df
        return self

    # ---------- SALVAR ----------
    def save(self) -> "DataPreprocessor":
        assert self.df_tr is not None
        self.df_tr.to_csv(self.cfg.output_path, index=False)
        return self

    # ---------- API DE ALTO NIVEL ----------
    def run(self) -> Tuple[pd.DataFrame, pd.Series]:
        (self.load()
             .engineer_features()
             .encode_categoricals()
             .transform_numericals()
             .save())
        return self.df_tr.copy(), self.groups.loc[self.df_tr.index].copy()
