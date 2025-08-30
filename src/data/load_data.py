import pandas as pd
import os
from pprint import pprint

def load_csv_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:

        df = pd.read_csv(file_path)

        pprint(df.head())
        
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Define the path to the data file
    data_path = os.path.join("data", "processed", "processed.csv")
    
    # load_csv_data(data_path)