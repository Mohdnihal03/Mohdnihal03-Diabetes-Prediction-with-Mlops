import pandas as pd
import argparse
import yaml
from get_data import get_data_load, read_configuration  # Updated import names
import os
from sklearn.model_selection import train_test_split

def split_and_Save(config_file_path):
    config = read_configuration(config_file_path)
    train_data_path  = config["split_data"]["train_Data_location"]
    test_data_path  = config["split_data"]["test_Data_location"]
    raw_data_path = config["load_data"]["load_Data_location"]
    print(raw_data_path)
    random_state_42 = config['base']['random_state']
    split_ratio = config['split_data']['test_size']
    df = pd.read_csv(raw_data_path , sep=',')
    
    train,test = train_test_split(df,test_size=split_ratio,random_state=random_state_42)
    train.to_csv(train_data_path,sep=',',index=False,encoding='utf-8')
    test.to_csv(test_data_path,sep=',',index=False,encoding='utf-8')

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Preprocess and save data based on configuration file")
    
    # Add an argument for the path to the YAML configuration file
    parser.add_argument("--config", default='params.yaml', help="Path to the YAML configuration file")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run the data preprocessing and saving function
    output=split_and_Save(config_file_path=args.config)
    print(output)
