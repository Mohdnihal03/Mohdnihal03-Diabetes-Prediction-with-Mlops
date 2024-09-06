import pandas as pd
import argparse
from get_data import get_data_load, read_configuration  # Updated import names
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import numpy as np
import os
import yaml
import mlflow
import mlflow.sklearn  # To log the model with MLflow
from urllib.parse import urlparse


def train_and_eval(config_file_path):
    config = read_configuration(config_file_path)

    # Extract configuration parameters
    Target = config['base']['target_column']
    model_dir = config['model_path']

    penalty = config['Algorithm']['LogisticRegression']['parameters']['penalty']
    C = config['Algorithm']['LogisticRegression']['parameters']['C']
    solver = config['Algorithm']['LogisticRegression']['parameters']['solver']
    max_iter = config['Algorithm']['LogisticRegression']['parameters']['max_iter']

    # Load training and testing data
    train = pd.read_csv('data/processed/train.csv', sep=',')
    test = pd.read_csv('data/processed/test.csv', sep=',')
    x_train = train.drop(Target, axis=1)
    x_test = test.drop(Target, axis=1)
    y_train = train[Target]
    y_test = test[Target]

    # MLflow configuration
    remote_server_uri = config['MLflow_config']['remote_server_uri']
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(config['MLflow_config']['experiment_title'])

    with mlflow.start_run(run_name=config['MLflow_config']['execution_name']) as mlops_run:
        # Initialize and train the LogisticRegression model
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
        model.fit(x_train, y_train)

        # Make predictions
        y_pred = model.predict(x_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param('penalty', penalty)
        mlflow.log_param('C', C)
        mlflow.log_param('solver', solver)
        mlflow.log_param('max_iter', max_iter)

        # Log metrics
        mlflow.log_metric('Mean Squared Error', mse)
        mlflow.log_metric('Mean Absolute Error', mae)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('r2', r2)

        # Log the model
        model_path = os.path.join(model_dir, "LogisticRegressionModel.joblib")
        joblib.dump(model, model_path)

        # Check if the artifact store is local and log the model accordingly
        tracking_uri_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_uri_type_store == 'file':
            # Log and register the model if the artifact URI is a file
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=config['MLflow_config']['model_registry_name'])
        else:
            # Log the model without registering if the artifact store is remote
            mlflow.sklearn.log_model(model, artifact_path="model")

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate the model based on configuration file")

    # Add an argument for the path to the YAML configuration file
    parser.add_argument("--config", default='params.yaml', help="Path to the YAML configuration file")

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the train and evaluation function
    train_and_eval(config_file_path=args.config)
