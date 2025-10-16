from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

MODEL_CLASSES = {
    "DecisionTree": DecisionTreeRegressor,
    'RandomForest': RandomForestRegressor,
    'HGB Regressor': HistGradientBoostingRegressor,
    'XGBoost': xgb.XGBRegressor, 
    'SVR': SVR,
}

# Para modelos Multi-Output, este es el MultiOutputRegressor CLASE.
MODEL_MAPPER = {
    "RandomForest": RandomForestRegressor,      # Soporte nativo
    "XGBoost": MultiOutputRegressor,             # Usamos la CLASE MultiOutputRegressor
}