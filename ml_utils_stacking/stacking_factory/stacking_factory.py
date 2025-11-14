from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingRegressor
from .models_registry import MODEL_CLASSES


class StackingModelFactory(BaseEstimator, RegressorMixin):
    """
    Estimator compatible con sklearn que construye un StackingRegressor
    usando clases declaradas en MODEL_CLASSES.
    """

    def __init__(self, base_models=None, final_model=None):
        # No construimos modelos aquí. Solo guardamos configs.
        self.base_models = base_models or {}
        self.final_model = final_model or {}

        self.model_ = None   # El modelo real se construirá en fit()

    def _build_model(self):
        """Construye realmente el stacking cuando fit() es llamado."""

        estimators = []
        for name, cfg in self.base_models.items():
            ModelClass = MODEL_CLASSES[cfg["model"]]
            estimators.append((name, ModelClass(**cfg.get("params", {}))))

        FinalClass = MODEL_CLASSES[self.final_model["model"]]
        final_estimator = FinalClass(**self.final_model.get("params", {}))

        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator
        )

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    # ---------- sklearn compatibility ----------
    def get_params(self, deep=True):
        return {
            "base_models": self.base_models,
            "final_model": self.final_model
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self
