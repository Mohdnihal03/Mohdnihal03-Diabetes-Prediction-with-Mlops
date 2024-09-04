import pandas as pd
import argparse
from get_data import get_data_load, read_configuration  # Updated import names
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix , r2_score , mean_squared_error , mean_absolute_error
import joblib
import json
import numpy as np
import os 
import yaml
import mlflow
from urllib.parse import urlparse # parse method from one file to another file

def train_and_eval(config_file_path):
    config = read_configuration(config_file_path)

    Target = config['base']['target_column']
    model_dir = config['model_path']

    penalty = config['Algorithm']['LogisticRegression']['parameters']['penalty']
    C = config['Algorithm']['LogisticRegression']['parameters']['C']
    solver = config['Algorithm']['LogisticRegression']['parameters']['solver']
    max_iter = config['Algorithm']['LogisticRegression']['parameters']['max_iter']

    train = pd.read_csv('data/processed/train.csv',sep=',')
    test = pd.read_csv('data/processed/test.csv',sep=',')
    x_train = train.drop(Target,axis=1)
    x_test = test.drop(Target,axis=1)
    y_train = train[Target]
    y_test = test[Target]

    model = LogisticRegression(penalty=penalty,C=C,solver=solver,max_iter=max_iter)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    conf_matrix = confusion_matrix(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    # print('Mean squared error:%s' %mse)
    # print('Mean Absolute error:%s' %mae)
    # print('accuracy:%s' %accuracy)
    # print('confusion_matrix:%s' %conf_matrix)
    # print('r2:%s' %r2)


    score_file = config['reports']['scores']
    params_file = config['reports']['params']


    with open(score_file,'w') as f:
        scores = {
            'Mean Squared Error':mse,
            'Mean Absolute Error':mae,
            'Accuracy of model':accuracy,
            # 'Confusion matrix':conf_matrix,
            'R2 score ': r2
        }
        json.dump(scores, f, indent=4)


    with open(params_file,'w') as file:
        params = {
            'penalty':penalty,
            'C':C,
            'solver':solver,
            'max_iter': max_iter
        }
        json.dump(params, file, indent=4)

    os.makedirs(model_dir,exist_ok=True)
    model_path = os.path.join(model_dir,"LogisticRegressionModel.joblib")
    joblib.dump(model,model_path)










if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Split and save data based on configuration file")
    
    # Add an argument for the path to the YAML configuration file
    parser.add_argument("--config", default='params.yaml', help="Path to the YAML configuration file")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run the data splitting and saving function
    train_and_eval(config_file_path=args.config)