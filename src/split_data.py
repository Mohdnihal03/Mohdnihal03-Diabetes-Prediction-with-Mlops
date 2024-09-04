import pandas as pd
import argparse
from get_data import get_data_load, read_configuration  # Updated import names
from sklearn.model_selection import train_test_split

def split_and_save_data(config_file_path):
    """
    Loads raw data, splits it into training and testing sets, and saves them to specified locations.
    
    Parameters:
    - config_file_path (str): Path to the YAML configuration file.
    
    Returns:
    - None
    """
    # Load configuration settings from the YAML file
    config = read_configuration(config_file_path)
    
    # Extract relevant paths and parameters from the configuration file
    train_data_path = config["split_data"]["train_Data_location"]
    test_data_path = config["split_data"]["test_Data_location"]
    raw_data_path = config["load_data"]["load_Data_location"]
    test_size_ratio = config["split_data"]["test_size"]
    random_state = config['base']['random_state']
    
    # Load the raw data from the specified CSV file
    df = pd.read_csv(raw_data_path, sep=',')
    
    # Split the data into training and testing sets
    train, test = train_test_split(df, test_size=test_size_ratio, random_state=random_state)
    
    # Save the training and testing sets to their respective locations
    train.to_csv(train_data_path, sep=',', index=False, encoding='utf-8')
    test.to_csv(test_data_path, sep=',', index=False, encoding='utf-8')

    # Optionally, return a confirmation message or the paths where data was saved
    return train_data_path, test_data_path

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Split and save data based on configuration file")
    
    # Add an argument for the path to the YAML configuration file
    parser.add_argument("--config", default='params.yaml', help="Path to the YAML configuration file")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run the data splitting and saving function
    train_path, test_path = split_and_save_data(config_file_path=args.config)
    
    # Optionally print the paths where the data was saved
    print(f"Training data saved to: {train_path}")
    print(f"Testing data saved to: {test_path}")
