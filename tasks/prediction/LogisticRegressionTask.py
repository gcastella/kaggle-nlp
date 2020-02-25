import pandas as pd
from sklearn.linear_model import LogisticRegression
from .BasePredictionTask import BasePredictionTask


class LogisticRegressionTask(BasePredictionTask):
    def fit_model(self, df: pd.DataFrame, target_var="target"):
        X_train = df.drop("target")
        y_train = df["target"]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def model_predictions(self, df: pd.DataFrame, model) -> pd.DataFrame:
        return model.predict(df)
