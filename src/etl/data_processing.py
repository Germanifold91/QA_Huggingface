"""Data processing functions for converting Markdown files to text and creating a master dataset."""

from typing import Dict, Any
from bs4 import BeautifulSoup
from datasets import Dataset
import pandas as pd
import os
import markdown
import yaml


def create_file_table(
    folder_path: str, destination_path: str = None, save: bool = False
) -> pd.DataFrame:
    """
    Creates a DataFrame containing the file paths of all the Markdown files (.md)
    found in the specified folder path.

    Args:
        folder_path (str): The path to the folder containing the Markdown files.
        destination_path (str, optional): The path to save the DataFrame as a CSV file.
            Defaults to None.
        save (bool, optional): Whether to save the DataFrame as a CSV file.
            Defaults to False.

    Raises:
        ValueError: If the provided folder path does not exist.
        ValueError: If save is True but destination_path is not provided.
        IOError: If failed to save the DataFrame as a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the file paths.

    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path does not exist: {folder_path}")

    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                try:
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    file_table = pd.DataFrame({"FILE_PATH": file_paths})

    if save:
        if destination_path is None:
            raise ValueError("Destination path must be provided if save is True.")
        try:
            file_table.to_csv(destination_path)
        except Exception as e:
            raise IOError(f"Failed to save the DataFrame: {e}")

    return file_table


def save_text_to_file(text_content: str, output_path: str) -> None:
    """
    Save the given text content to a file at the specified output path.

    Args:
        text_content (str): The text content to be saved.
        output_path (str): The path where the file will be saved.

    Returns:
        None
    """
    with open(output_path, "w", encoding="utf-8") as text_file:
        text_file.write(text_content)


def markdown_file_to_text_and_save(md_file_path: str, output_dir: str) -> str:
    """
    Converts a Markdown file to plain text, saves it to a new file, and returns the path of the new file.

    Args:
        md_file_path (str): The path to the Markdown file.
        output_dir (str): The directory where the output file will be saved.

    Returns:
        str: The path of the new text file.

    Raises:
        Exception: If there is an error while processing or saving the file.
    """
    try:
        # Read the contents of the Markdown file
        with open(md_file_path, "r", encoding="utf-8") as file:
            md_content = file.read()

        # Convert Markdown to HTML
        html_content = markdown.markdown(md_content)

        # Extract plain text from HTML
        text_content = BeautifulSoup(html_content, "html.parser").get_text(
            separator="\n"
        )

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate the new file name by replacing the extension with '.txt'
        base_name = os.path.basename(md_file_path)
        new_file_name = os.path.splitext(base_name)[0] + ".txt"
        text_file_path = os.path.join(output_dir, new_file_name)

        # Save the text content to the output file
        save_text_to_file(text_content, text_file_path)

        return text_file_path
    except Exception as e:
        raise Exception(f"Failed to process and save file due to: {e}")


def process_and_update_df(
    master_df: pd.DataFrame, output_dir: str, save: bool, saving_path: str
) -> pd.DataFrame:
    """
    Process and update the given DataFrame by applying a conversion and saving function to each row.

    Args:
        master_df (pd.DataFrame): The DataFrame to be processed and updated.
        output_dir (str): The directory where the converted files will be saved.
        save (bool): A flag indicating whether to save the updated DataFrame.
        saving_path (str): The path to save the updated DataFrame.

    Returns:
        pd.DataFrame: The processed and updated DataFrame.

    Raises:
        IOError: If there is an error while saving the updated DataFrame.
    """
    # Apply the conversion and saving function to each row and store the new file paths
    master_df["TEXT_FILE_PATH"] = master_df["FILE_PATH"].apply(
        lambda x: markdown_file_to_text_and_save(x, output_dir)
    )

    if save:
        try:
            master_df.to_csv(saving_path, index=False)
        except Exception as e:
            raise IOError(f"Failed to save the updated DataFrame: {e}")

    return master_df


def arrow_dataset(
    master_df: pd.DataFrame,
    file_path_column: str,
    text_column_name: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Process the data in the master DataFrame by reading text content from files specified in the file_path_column.
    The text content is then stored in the text_column_name column of the DataFrame.
    Finally, the updated DataFrame is converted to a Dataset and saved to the specified directory.

    Args:
        master_df (pd.DataFrame): The master DataFrame containing the data to be processed.
        file_path_column (str): The name of the column in master_df that contains the file paths.
        text_column_name (str): The name of the column in master_df where the text content will be stored.
        save_path (str): The directory where the Dataset will be saved.

    Returns:
        pd.DataFrame: The updated master DataFrame with the text content added.

    Raises:
        FileNotFoundError: If a file specified in the file_path_column does not exist.
    """
    # Check if the text_column_name already exists, if not, initialize it
    if text_column_name not in master_df.columns:
        master_df[text_column_name] = pd.Series(dtype="str")

    # Iterate through the DataFrame and read the text content from each file
    for index, row in master_df.iterrows():
        file_path = row[file_path_column]
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text_content = file.read()
            master_df.at[index, text_column_name] = text_content
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    # Convert the updated DataFrame to a Dataset
    dataset = Dataset.from_pandas(master_df)

    # Save the Dataset to the specified directory
    dataset.save_to_disk(save_path)

    return master_df


def generate_master(de_params: Dict[str, Any]) -> None:
    """
    Generate the master dataset by performing a series of steps:
    1. Load parameters from the dictionary.
    2. Create a CSV file that lists paths to individual text files.
    3. Update the CSV file with the actual text content of the files.
    4. Convert the updated CSV file into a Hugging Face Arrow dataset.

    Args:
        de_params (Dict[str,Any]): A dictionary containing the necessary parameters.

    Returns:
        None
    """
    print("Loading Params")
    # Read parameters from the dictionary
    documents_paths = de_params["documents_path"]
    save_csv = de_params["save_csv"]
    csv_path = de_params["csvdf_path"]
    text_files_path = de_params["text_files_path"]
    arrow_path = de_params["arrowdf_path"]

    print(f"Creating CSV table and saving it as {csv_path}")
    # Step 2: Create a CSV file that lists paths to individual text files
    file_table = create_file_table(
        folder_path=documents_paths, destination_path=csv_path, save=save_csv
    )

    print(f"Updating CSV table and saving it as {csv_path}")
    # Step 3: Update the CSV file with the actual text content of the files
    updated_table = process_and_update_df(
        master_df=file_table,
        output_dir=text_files_path,
        save=save_csv,
        saving_path=csv_path,
    )

    print(f"Creating and saving arrow dataset to {arrow_path}")
    # Step 4: Convert the updated CSV file into a Hugging Face Arrow dataset
    arrow_dataset(
        master_df=updated_table,
        file_path_column="TEXT_FILE_PATH",
        text_column_name="TEXT",
        save_path=arrow_path,
    )


if __name__ == "__main__":

    # Load the YAML configuration file
    with open("conf/local.yml", "r") as yamlfile:
        config = yaml.safe_load(
            yamlfile
        )  # Use safe_load to load the YAML configuration file safely

    de_params = config.get("data_processing_params")
    if de_params is None:
        raise ValueError(
            "Invalid YAML configuration file. Missing 'data_processing_params' section."
        )

    generate_master(de_params)
