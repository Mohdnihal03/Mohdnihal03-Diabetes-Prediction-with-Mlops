import pandas as pd
import argparse
import yaml
from get_data import get_data_load, read_configuration  # Updated import names
import os

def preprocess_and_save_data(config_file_path):
    """
    Loads data from the source, preprocesses it by renaming columns,
    and saves it to a specified location.
    
    Parameters:
    - config_file (str): Path to the YAML configuration file.
    
    Returns:
    - None
    """
    # Load configuration settings from the YAML file
    config = read_configuration(config_file_path)
    
    # Load the data using the configuration settings
    df = get_data_load(config_file_path)
    
    # Rename columns by replacing spaces with underscores
    renamed_columns = [col.replace(" ", "_") for col in df.columns]
    
    # Get the path to save the preprocessed data
    save_path = config["load_data"]["load_Data_location"]
    
    # Save the preprocessed data to a CSV file
    df.to_csv(save_path, sep=',', index=False, header=renamed_columns)
    
    # Optionally return a confirmation message or the path where data is saved
    return save_path

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Preprocess and save data based on configuration file")
    
    # Add an argument for the path to the YAML configuration file
    parser.add_argument("--config", default='params.yaml', help="Path to the YAML configuration file")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run the data preprocessing and saving function
    saved_path = preprocess_and_save_data(config_file_path=args.config)
    
    # Optionally print the path where the data was saved
    print(f"Data saved to: {saved_path}")
