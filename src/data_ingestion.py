import os  # Import os module for filesystem operations like joining paths and creating directories
import logging  # Import logging module to record events, errors, and debug info during execution
import pandas as pd  # Import pandas for reading and manipulating tabular data (DataFrames)
from sklearn.model_selection import train_test_split  # Import function to split data into train/test sets


# logging configure
logger = logging.getLogger('data_ingestion')  # Create a logger object named 'data_ingestion' for this module
logger.setLevel('DEBUG')  # Set the logger's overall threshold to DEBUG (captures all levels: DEBUG and above)

console_handler = logging.StreamHandler()  # Create a handler that sends log output to the console (terminal)
console_handler.setLevel('DEBUG')  # Console will show DEBUG-level messages and above (everything)

file_handler = logging.FileHandler('errors.log')  # Create a handler that writes log output to a file named errors.log
file_handler.setLevel('ERROR')  # File will only capture ERROR-level messages and above (keeps file focused on real issues)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Define the log message format: timestamp, logger name, level, message
console_handler.setFormatter(formatter)  # Apply the format to the console handler
file_handler.setFormatter(formatter)  # Apply the same format to the file handler

logger.addHandler(console_handler)  # Attach the console handler to the logger so it can output there
logger.addHandler(file_handler)  # Attach the file handler to the logger so it can output there


# Function to get the raw data
def load_data(data_url: str) -> pd.DataFrame:  # Define function that takes a URL/path string and returns a DataFrame
    """Load raw payment transaction data from a URL or local path."""  # Docstring explaining the function's purpose
    try:  # Start a try block to catch errors that may occur while reading the file
        df = pd.read_csv(data_url)  # Read the CSV data from a link or folder into a DataFrame
        logger.debug("Data loaded successfully from %s", data_url)  # Log a debug message confirming successful load
        return df  # Return the loaded DataFrame to the caller
    except pd.errors.ParserError as e:  # Catch errors specifically caused by malformed/broken CSV content
        logger.error("Failed to parse the CSV file from %s.", data_url)  # Log an error describing which file failed to parse
        logger.error(e)  # Log the exact parser error message/details
        raise  # Re-raise the exception to stop the pipeline instead of continuing silently
    except Exception as e:  # Catch any other unexpected error (e.g. network issue, missing file)
        logger.error("An unexpected error occurred while loading the data.")  # Log a generic error message for context
        logger.error(e)  # Log the exact exception details
        raise  # Re-raise the exception to halt execution and avoid using blank/broken data


# Defining a light validation step — heavy cleaning/feature engineering lives in
# data_preprocessing.py, so data/raw stays genuinely raw
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:  # Define function taking a raw DataFrame and returning a validated one
    """Do minimal sanity checks and drop exact duplicate rows before splitting."""  # Docstring explaining the function's purpose
    try:  # Start a try block to catch errors during validation/cleaning
        # Work on a copy to avoid SettingWithCopyWarning on the original frame
        final_df = df.copy()  # Create a copy of the input DataFrame so the original isn't mutated

        # Guard against a schema change wiping out the target column entirely
        if "is_fraud" not in final_df.columns:  # Check whether the required target column exists
            raise KeyError("is_fraud")  # Manually raise a KeyError if the column is missing

        # Drop fully duplicated rows so the same transaction isn't split across
        # both train and test
        final_df = final_df.drop_duplicates()  # Remove exact duplicate rows from the DataFrame

        logger.debug("Data preprocessing completed successfully.")  # Log a debug message confirming preprocessing succeeded
        return final_df  # Return the cleaned/validated DataFrame
    except KeyError as e:  # Catch the case where an expected column is missing
        logger.error("Missing expected column %s in the dataframe.", e)  # Log which column was missing
        raise  # Re-raise to crash the pipeline and prevent training on a corrupted schema
    except Exception as e:  # Catch any other unforeseen error during preprocessing
        logger.error("An unexpected error occurred during preprocessing.\n%s", e)  # Log the unexpected error's details
        raise  # Re-raise to stop further execution of the pipeline


# Save the train and test data to a folder
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:  # Define function to save train/test splits to disk
    """Save train and test splits into a local folder destination."""  # Docstring explaining the function's purpose
    try:  # Start a try block to catch errors during file saving
        target_dir = os.path.join(data_path, "raw")  # Build the target directory path (e.g. data/raw)
        os.makedirs(target_dir, exist_ok=True)  # Create the directory if it doesn't already exist (no error if it does)

        # Save files as train and test split without dumping the pandas index column
        train_data.to_csv(os.path.join(target_dir, "train.csv"), index=False)  # Write the training DataFrame to train.csv without the index column
        test_data.to_csv(os.path.join(target_dir, "test.csv"), index=False)  # Write the testing DataFrame to test.csv without the index column

        logger.debug("Train and test data saved successfully to %s", target_dir)  # Log a debug message confirming save succeeded

    except Exception as e:  # Catch any error that occurs while saving the files
        logger.error("An unexpected error occurred while saving the data.\n%s", e)  # Log the details of the save failure
        raise  # Re-raise to prevent downstream steps from reading incomplete/missing datasets


def main():  # Define the main entry-point function that orchestrates the full ingestion pipeline
    try:  # Start a try block to catch any error across the whole pipeline
        # Fetch the remote source data path
        url = "https://raw.githubusercontent.com/Donatus-Victor/fraud_detection/refs/heads/main/payment_transactions.csv"  # Define the URL of the raw source CSV
        df = load_data(url)  # Call load_data to fetch the raw data into a DataFrame

        # Validate, split, and save data splits
        final_df = preprocess_data(df)  # Call preprocess_data to validate and clean the raw DataFrame
        train_data, test_data = train_test_split(  # Split the cleaned data into training and testing sets
            final_df, test_size=0.2, stratify=final_df["is_fraud"], random_state=2  # Use 20% for test, stratify by target class, fixed seed for reproducibility
        )
        save_data(train_data, test_data, data_path="data")  # Save both splits to the "data/raw" folder

        logger.debug("Data ingestion and processing completed successfully.")  # Log a debug message confirming the whole pipeline succeeded
    except Exception as e:  # Catch any exception that propagated up from the pipeline steps
        logger.error("Failed to complete the data ingestion process: %s", e)  # Log the failure with the exception details


# Prevent code from running automatically if this file is imported as a modular helper tool elsewhere
if __name__ == "__main__":  # Check if this script is being run directly (not imported as a module)
    main()  # Call the main function to execute the pipeline