from src.utils.text_utils import clean_text
import pandas as pd
from typing import Dict, Any
from bs4 import BeautifulSoup
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