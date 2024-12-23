from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "elastic-net-regression"
EXPERIMENT_NAME = "elastic-net-trials"


class ElasticNetCustom(ElasticNet):
    """
    A custom random forest classifier.
    The RandomForestClassifier class is extended by adding a callback function within its fit method.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        super().fit(X, y)
        # if a "callback" key is passed, call the "callback" function by passing the fitted estimator
        if "callback" in kwargs:
            kwargs["callback"](self)
        return self


class Logger:
    """
    Logger class stores the test dataset,
    and logs sklearn random forest estimator in rf_logger method.
    """

    def __init__(
        self, experiment_id: str | None, X_test: np.ndarray, y_test: np.ndarray
    ):
        self.X_test = X_test
        self.y_test = y_test
        self.experiment_id = experiment_id

    def log(self, model: ElasticNet):
        # log the model in the nested mlflow runs
        uuid = str(uuid4())
        run_name = f"alpha={model.alpha},l1_ratio={model.l1_ratio}-{uuid[0:8]}"
        with mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            nested=True,
            log_system_metrics=True,
        ):
            mlflow.log_param("alpha", model.alpha)
            mlflow.log_param("l1_ratio", model.l1_ratio)

            y_pred = model.predict(self.X_test)
            signature = infer_signature(self.X_test, y_pred)
            metrics = calculate_metrics(y_pred, self.y_test)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                model, "model", signature=signature, input_example=self.X_test
            )
        return None


def calculate_metrics(y_pred: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    return metrics


def score_fold(
    model: ElasticNet, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_pred, y_test)
    return metrics


def main():
    data = load_iris()
    Xy = pd.DataFrame(data.data, columns=data.feature_names)
    target_name = "sepal length (cm)"
    features_names = ["sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    X = Xy.loc[:, features_names]
    y = Xy.loc[:, target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = ElasticNetCustom(random_state=9)
    param_grid = {"alpha": [0.5, 1.0], "l1_ratio": [0.5, 1.0]}
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid = GridSearchCV(model, param_grid, cv=3, scoring=scorer)

    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment_id:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"version": "v1", "priority": "P1"},
        )
    logger = Logger(experiment_id, X_test, y_test)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="grid_search",
        log_system_metrics=True,
    ):
        dataset = mlflow.data.from_pandas(Xy, name="iris", targets=target_name)
        mlflow.log_input(dataset, context="train")
        grid.fit(X_train, y_train, callback=logger.log)

        # log the best estimator fround by grid search in the outer mlflow run
        mlflow.log_param("alpha", grid.best_params_["alpha"])
        mlflow.log_param("l1_ratio", grid.best_params_["l1_ratio"])

        y_pred = grid.predict(X_test)
        signature = infer_signature(X_test, y_pred)
        metrics = calculate_metrics(y_pred, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            grid.best_estimator_,
            "model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

    # Get some info about the logged model
    last_run = mlflow.last_active_run()
    run_id = last_run.info.run_id
    print(f"Logged data and model in run: {run_id}")
    print(f"Model saved in run {last_run.info.run_uuid}")


if __name__ == "__main__":
    main()
