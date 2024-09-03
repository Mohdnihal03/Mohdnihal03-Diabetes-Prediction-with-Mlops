import pandas as pd 
import argparse  
import yaml  

# Function to load data from a file specified in the configuration
def get_data_load(config_file_path):
    # Read the configuration settings from the YAML file
    config = read_configuration(config_file_path)
    # Get the path to the data file from the configuration
    data_file_path = config["load_data"]["load_Data_location"]
    # Load the data from the CSV file into a DataFrame
    data_frame = pd.read_csv(data_file_path, sep=',', encoding='utf-8')
    return data_frame

# Function to read configuration settings from a YAML file
def read_configuration(config_file_path):
    # Open the YAML file and load its contents
    with open(config_file_path) as yaml_file:
        configuration = yaml.safe_load(yaml_file)
    return configuration

# This block of code runs when the script is executed directly
if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Load data from a configuration file")
    # Add an argument to specify the path to the configuration file
    parser.add_argument("--config", default="params.yaml", help="Path to the YAML configuration file")
    # Parse the command-line arguments
    args = parser.parse_args()
    # Load the data using the specified configuration file
    data = get_data_load(config_file_path=args.config)
    # Optionally print the first few rows of the data to verify it was loaded correctly
    # print(data)
