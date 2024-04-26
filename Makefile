.PHONY: create-env update-env activate-env data-processing data-training process-question clean

# Define variables
CONDA_ENV_NAME=huggingface_env
QUESTION=Your question here

# Create Conda environment
create-env:
	conda create --name $(CONDA_ENV_NAME) python=3.10 -y
	conda activate $(CONDA_ENV_NAME)
	conda install --file requirements.txt -y

# Update Conda environment
update-env:
	conda activate $(CONDA_ENV_NAME)
	conda install --file requirements.txt -y

# Activate Conda environment
activate-env:
	@echo "Activating environment..."
	@echo "Run 'conda activate $(CONDA_ENV_NAME)'"

# Run data processing
data-processing:
	@echo "Running data processing script..."
	conda run -n $(CONDA_ENV_NAME) python src/etl/data_processing.py

# Generate question answering training data
data-training:
	@echo "Running data training generation script..."
	conda run -n $(CONDA_ENV_NAME) python src/etl/data_training.py

# Fine tune model with training data
tune-model:
	@echo "Running model fine tuning script..."
	conda run -n $(CONDA_ENV_NAME) python src/model_tuning/fine_tuning.py

# Run question answering pipeline
process-question:
	conda run -n $(CONDA_ENV_NAME) python src/features/qa_pipe.py --question "$(QUESTION)"

# Clean Conda environment
clean:
	conda remove --name $(CONDA_ENV_NAME) --all -y