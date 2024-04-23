from typing import Dict
import pandas as pd
import json

def track_execution(qa_pipe_output: Dict[str, any], 
                    output_file: str, 
                    json_file: str) -> None:
    """
    Track the execution of the QA pipe output and save the tracking data as a CSV file.

    Args:
        qa_pipe_output (List[Dict[str, any]]): The output of the QA pipe.
        output_file (str): The path to save the tracking data as a CSV file.
        json_file (str): The path to save the tracking data as a JSON file.

    Returns:
        None
    """
    # Check if the tracking file already exists
    try:
        tracking_df = pd.read_csv(output_file)
    except FileNotFoundError:
        # Create an empty DataFrame if the file doesn't exist
        print("Tracking file not found. Creating a new one.")
        tracking_df = pd.DataFrame()

    # Create a list to store the tracking data
    tracking_data = []

    # Create a dictionary with the tracking data
    tracking_entry = {
        'question': qa_pipe_output['question'],
        'answer': qa_pipe_output['answer'],
        'document_path': qa_pipe_output['document_path'],
        'score': qa_pipe_output['score'],
        'start': qa_pipe_output['start'],
        'end': qa_pipe_output['end']
    }

    # Append the tracking entry to the list
    tracking_data.append(tracking_entry)

    # Create a pandas DataFrame from the tracking data
    new_tracking_df = pd.DataFrame(tracking_data)

    # Append the new tracking data to the existing DataFrame
    tracking_df = pd.concat([tracking_df, new_tracking_df], ignore_index=True)

    # Save the DataFrame as a CSV file
    tracking_df.to_csv(output_file, index=False)

    # Convert the tracking data to a JSON object
    tracking_json = json.dumps(tracking_data)

    # Check if the JSON file already exists
    try:
        with open(json_file, 'r') as f:
            existing_json = json.load(f)
    except FileNotFoundError:
        # Create an empty list if the file doesn't exist
        print("JSON file not found. Creating a new one.")
        existing_json = []

    # Append the new tracking data to the existing JSON object
    existing_json.extend(tracking_data)

    # Save the JSON object to the file
    with open(json_file, 'w') as f:
        json.dump(existing_json, f)

    print("Tracking data saved successfully.")