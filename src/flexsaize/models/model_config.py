from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

from .stacking_factory import make_stacking_v2



MODEL_CLASSES = {
    "DecisionTree": DecisionTreeRegressor,
    'RandomForest': RandomForestRegressor,
    'HGB Regressor': HistGradientBoostingRegressor,
    'XGBoost': xgb.XGBRegressor, 
    'SVR': SVR,
    "StackingV2": make_stacking_v2
}

# Para modelos Multi-Output, este es el MultiOutputRegressor CLASE.
MODEL_MAPPER = {
    "DecisionTree": MultiOutputRegressor,
    "RandomForest": RandomForestRegressor,              # Soporte multi-output nativo
    "HGB Regressor": HistGradientBoostingRegressor,             # Soporte multi-output nativo
    "XGBoost": MultiOutputRegressor,   # Necesita wrapper
    "SVR": MultiOutputRegressor, 
    "StackingV2": make_stacking_v2
}