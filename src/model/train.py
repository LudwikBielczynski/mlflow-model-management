import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:5000")


def main():
    # Prepare data
    data = load_iris()
    Xy = pd.DataFrame(data.data, columns=data.feature_names)
    target_name = "sepal length (cm)"
    features_names = ["sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    X = Xy.loc[:, features_names]
    y = Xy.loc[:, target_name]

    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

    dataset = mlflow.data.from_pandas(Xy, name="iris", targets=target_name)
    mlflow.log_input(dataset, context="train")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train a model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Score
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mlflow.log_metrics({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    # Register model
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X,
        registered_model_name="linear-regression",
    )

    # Get some info about the logged model
    last_run = mlflow.last_active_run()
    run_id = last_run.info.run_id
    print(f"Logged data and model in run: {run_id}")
    print(f"Model saved in run {last_run.info.run_uuid}")


if __name__ == "__main__":
    main()
