import numpy as np
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")


def main():
    mlflow.sklearn.autolog()

    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # train a model
    model = LinearRegression()
    model.fit(X, y)

    last_run = mlflow.last_active_run()
    run_id = last_run.info.run_id
    print(f"Logged data and model in run: {run_id}")
    print(f"Model saved in run {last_run.info.run_uuid}")


if __name__ == "__main__":
    main()
