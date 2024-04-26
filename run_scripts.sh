#!/bin/bash

# Run data_processing.py
python src/etl/data_processing.py

# Run qa_pipe.py with a question as input
python src/features/qa_pipe.py --question "Your question here"