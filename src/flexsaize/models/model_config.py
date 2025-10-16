from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

MODEL_MAPPER = {
    "DecisionTree": DecisionTreeRegressor,
    'RandomForest': RandomForestRegressor,
    'HGB Regressor': HistGradientBoostingRegressor,
    'XGBoost': MultiOutputRegressor(xgb.XGBRegressor()), 
    'SVR': MultiOutputRegressor(SVR()),
}