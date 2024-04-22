from src.utils.text_utils import clean_text
import pandas as pd
import os

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