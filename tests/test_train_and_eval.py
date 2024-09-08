# tests/test_train_and_eval.py

import pytest
import pandas as pd
import json
import os
from src.train_and_eval import train_and_eval

@pytest.fixture
def mock_config_file():
    mock_config_content = """
    base:
      target_column: Outcome

    data_source:
      source_Data_location: source_data/pima_indian.csv

    load_data:
      load_Data_location: data/raw/raw_Data.csv

    split_data:
      train_Data_location: data/processed/train.csv
      test_Data_location: data/processed/test.csv
      test_size: 0.2

    Algorithm:
      LogisticRegression:
        parameters:
          penalty: 'l1'
          C: 10
          solver: 'saga'
          max_iter: 75

    model_path: models

    MLflow_config:
      remote_server_uri: 'https://dagshub.com/mohammednihal9986/Mohdnihal03-Diabetes-Prediction-with-Mlops.mlflow'
      storage_location: artifacts
      experiment_title: LogisticRegression_Mlops
      execution_name: mlops-diabetes
      model_registry_name: LogisticRegression-diabetes-prediction

    reports:
      params: reports/params.json
      scores: reports/scores.json
    """
    with open('tests/mock_params.yaml', 'w') as f:
        f.write(mock_config_content)
    yield 'tests/mock_params.yaml'
    os.remove('tests/mock_params.yaml')
    for file in ['data/processed/train.csv', 'data/processed/test.csv', 'models/LogisticRegressionModel.joblib']:
        if os.path.exists(file):
            os.remove(file)

def test_train_and_eval(mock_config_file):
    # Ensure directories exist for saving files
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Run the function
    train_and_eval(config_file_path=mock_config_file)

    # Check if files are created
    assert os.path.exists('models/LogisticRegressionModel.joblib')
    assert os.path.exists('reports/params.json')
    assert os.path.exists('reports/scores.json')

    # Validate content of reports
    with open('reports/params.json', 'r') as f:
        params = json.load(f)
        assert 'penalty' in params
        assert 'C' in params
        assert 'solver' in params
        assert 'max_iter' in params

    with open('reports/scores.json', 'r') as f:
        scores = json.load(f)
        assert 'Mean Squared Error' in scores
        assert 'Mean Absolute Error' in scores
        assert 'Accuracy of model' in scores
        assert 'R2 score ' in scores
