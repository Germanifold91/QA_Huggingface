from typing import Dict
import pandas as pd

def track_execution(qa_pipe_output: Dict[str, any], 
                    output_file: str) -> None:
    """
    Track the execution of the QA pipe output and save the tracking data as a CSV file.

    Args:
        qa_pipe_output (List[Dict[str, any]]): The output of the QA pipe.
        output_file (str): The path to save the tracking data as a CSV file.

    Returns:
        None
    """
    # Check if the tracking file already exists
    try:
        tracking_df = pd.read_csv(output_file)
    except FileNotFoundError:
        # Create an empty DataFrame if the file doesn't exist
        tracking_df = pd.DataFrame()

    # Create a list to store the tracking data
    tracking_data = []

    # Create a dictionary with the tracking data
    tracking_entry = {
        'question': qa_pipe_output['question'],
        'answer': qa_pipe_output['answer'],
        'document_path': qa_pipe_output['document_path']
    }

    # Append the tracking entry to the list
    tracking_data.append(tracking_entry)

    # Create a pandas DataFrame from the tracking data
    new_tracking_df = pd.DataFrame(tracking_data)

    # Append the new tracking data to the existing DataFrame
    tracking_df = pd.concat([tracking_df, new_tracking_df], ignore_index=True)

    # Save the DataFrame as a CSV file
    tracking_df.to_csv(output_file, index=False)