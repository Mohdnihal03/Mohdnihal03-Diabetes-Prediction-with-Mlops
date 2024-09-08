from http import client
from get_data import read_configuration  # Custom function to read configuration from params.yaml
import argparse
import mlflow
from mlflow.tracking import MlflowClient  # MlflowClient is used for handling model registry operations
from pprint import pprint
import joblib
import os

def log_production_model(config_file_path):
    """
    Logs the best model (based on lowest mean absolute error) to MLflow model registry and promotes it to 'Production' stage.
    Loads the model and saves it locally for web app use.

    Parameters:
    config_path (str): Path to the YAML configuration file.
    """

    # Step 1: Load configuration settings
    config = read_configuration(config_file_path)
    mlflow_config = config["MLflow_config"]

    # Extract necessary parameters from the config file
    model_name = mlflow_config["model_registry_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Step 2: Set up MLflow tracking URI to the remote server
    mlflow.set_tracking_uri(remote_server_uri)

    # Step 3: Search for runs in experiment ID 1 and check the available columns
    runs = mlflow.search_runs([1])
    print("Available columns:", runs.columns)  # Print the available columns for inspection

    # If accuracy is not present, print metrics columns to see what is available
    metrics_columns = [col for col in runs.columns if col.startswith("metrics.")]
    print(f"Available metrics: {metrics_columns}")

    # Step 4: Modify this to use the correct metric name after inspecting
    # Let's assume the correct metric is 'metrics.accuracy', but adjust based on output
    if "metrics.accuracy" in metrics_columns:
        best = runs["metrics.accuracy"].max()  # Find the best accuracy score
        best_run_id = runs[runs["metrics.accuracy"] == best]["run_id"].iloc[0]  # Get the run ID with the highest accuracy
    else:
        raise KeyError("Accuracy metric is not found in the logged metrics.")

    # Step 5: Initialize MlflowClient to manage model versioning and stage transitions
    client = MlflowClient()

    # Step 6: Search for all model versions in the model registry and update stages
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        
        # If the run ID matches the one with the highest accuracy, promote it to 'Production'
        if mv["run_id"] == best_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]  # Get the path of the logged model
            pprint(mv, indent=4)  # Print the model version details

            # Promote the model version to 'Production' stage
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            # Any other versions are moved to 'Staging'
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )

    # Step 7: Load the 'Production' model from MLflow and save it locally for web app use
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_path = config["model_path"]  # Path to save the model locally for the web app
    joblib.dump(loaded_model, model_path)



if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Split and save data based on configuration file")
    
    # Add an argument for the path to the YAML configuration file
    parser.add_argument("--config", default='params.yaml', help="Path to the YAML configuration file")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run the data splitting and saving function
    log_production_model(config_file_path=args.config)
