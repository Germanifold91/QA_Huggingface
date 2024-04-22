from src.utils.text_utils import clean_text
from typing import Dict, Any
from bs4 import BeautifulSoup
from datasets import Dataset
import pandas as pd
import os
import markdown

def create_file_table(folder_path: str, 
                      destination_path: str = None, 
                      save: bool = False) -> pd.DataFrame:

    """
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path does not exist: {folder_path}")

    file_paths = []
    tokens_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                try:
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)

                    # Extract and clean tokens from the filename
                    file_name_without_ext = os.path.splitext(file)[0]
                    cleaned_tokens = clean_text(file_name_without_ext)
                    tokens_list.append(cleaned_tokens)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    file_table = pd.DataFrame({
        'FILE_PATH': file_paths,
        'TITLE_TOKENS': tokens_list  
    })
    
    if save:
        if destination_path is None:
            raise ValueError("Destination path must be provided if save is True.")
        try:
            file_table.to_csv(destination_path)
        except Exception as e:
            raise IOError(f"Failed to save the DataFrame: {e}")

    return file_table


def save_text_to_file(text_content: str, 
                      output_path: str) -> None:
    """Save text content to a specified path."""
    with open(output_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)


def markdown_file_to_text_and_save(md_file_path: str, output_dir: str) -> str:
    """
    """
    try:
        # Read the contents of the Markdown file
        with open(md_file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()

        # Convert Markdown to HTML
        html_content = markdown.markdown(md_content)

        # Extract plain text from HTML
        text_content = BeautifulSoup(html_content, 'html.parser').get_text(separator='\n')

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate the new file name by replacing the extension with '.txt'
        base_name = os.path.basename(md_file_path)
        new_file_name = os.path.splitext(base_name)[0] + '.txt'
        text_file_path = os.path.join(output_dir, new_file_name)

        # Save the text content to the output file
        save_text_to_file(text_content, text_file_path)

        return text_file_path
    except Exception as e:
        raise Exception(f"Failed to process and save file due to: {e}")
    

def process_and_update_df(master_df: pd.DataFrame, 
                          output_dir: str, 
                          save: bool, 
                          saving_path: str) -> pd.DataFrame:
    """

    """
    # Apply the conversion and saving function to each row and store the new file paths
    master_df['TEXT_FILE_PATH'] = master_df['FILE_PATH'].apply(lambda x: markdown_file_to_text_and_save(x, output_dir))

    if save:
        try:
            master_df.to_csv(saving_path, index=False)
        except Exception as e:
            raise IOError(f"Failed to save the updated DataFrame: {e}")
    
    return master_df


def arrow_dataset(master_df: pd.DataFrame, 
                  file_path_column: str, 
                  text_column_name:str, 
                  save_path: str) -> pd.DataFrame:
    """

    """
    # Check if the text_column_name already exists, if not, initialize it
    if text_column_name not in master_df.columns:
        master_df[text_column_name] = pd.Series(dtype='str')

    # Iterate through the DataFrame and read the text content from each file
    for index, row in master_df.iterrows():
        file_path = row[file_path_column]
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
            master_df.at[index, text_column_name] = text_content
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    # Convert the updated DataFrame to a Dataset
    dataset = Dataset.from_pandas(master_df)

    # Save the Dataset to the specified directory
    dataset.save_to_disk(save_path)

    return master_df


def generate_master(de_params: Dict[str,Any]) -> None:
    """

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
    file_table = create_file_table(folder_path=documents_paths, 
                                   destination_path=csv_path, 
                                   save=save_csv)
    
    print(f"Updating CSV table and saving it as {csv_path}")
    # Step 3: Update the CSV file with the actual text content of the files
    updated_table = process_and_update_df(master_df=file_table, 
                                          output_dir=text_files_path, 
                                          save=save_csv,
                                          saving_path=csv_path)
    
    print(f"Creating and saving arrow dataset to {arrow_path}")
    # Step 4: Convert the updated CSV file into a Hugging Face Arrow dataset
    arrow_dataset(master_df=updated_table,
                  file_path_column="TEXT_FILE_PATH",
                  text_column_name="TEXT",
                  save_path=arrow_path)