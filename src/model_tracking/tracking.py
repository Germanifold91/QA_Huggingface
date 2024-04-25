""" This module contains functions for tracking the execution of a question answering pipeline. """

from typing import Dict
import pandas as pd
import json


def track_execution(qa_pipe_output: Dict[str, any], json_file: str) -> None:
    """
    Track the execution of a question answering pipeline and save the tracking data to a JSON file.

    Args:
        qa_pipe_output (Dict[str, any]): The output of the question answering pipeline.
        json_file (str): The path to the JSON file where the tracking data will be saved.

    Returns:
        None
    """
    # Create a list to store the tracking data
    tracking_data = []

    # Create a dictionary with the tracking data
    tracking_entry = {
        "question": qa_pipe_output["question"],
        "answer": qa_pipe_output["answer"],
        "document_path": qa_pipe_output["document_path"],
        "score": qa_pipe_output["score"],
        "start": qa_pipe_output["start"],
        "end": qa_pipe_output["end"],
    }

    # Append the tracking entry to the list
    tracking_data.append(tracking_entry)

    # Check if the JSON file already exists
    try:
        with open(json_file, "r") as f:
            existing_json = json.load(f)
    except FileNotFoundError:
        # Create an empty list if the file doesn't exist
        print("JSON file not found. Creating a new one.")
        existing_json = []

    # Append the new tracking data to the existing JSON object
    existing_json.extend(tracking_data)

    # Save the JSON object to the file
    with open(json_file, "w") as f:
        json.dump(existing_json, f)

    print("Tracking data saved successfully.")
