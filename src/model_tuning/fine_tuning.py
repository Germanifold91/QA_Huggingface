""" Fine-tunes a pre-trained model for question answering using a dataset. """
from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering
from datasets import Dataset, load_from_disk
import yaml

def train_model(dataset, model_name, output_dir):
    """
    Trains a model for question answering using the provided dataset.

    Args:
        dataset (dict): A dictionary containing the training and validation datasets.
        model_name (str): The name or path of the pre-trained model to be fine-tuned.
        output_dir (str): The directory where the trained model will be saved.

    Returns:
        None
    """
    # Load the model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # The output directory
        num_train_epochs=3,  # The number of training epochs
        per_device_train_batch_size=16,  # The batch size for training
        per_device_eval_batch_size=64,  # The batch size for evaluation
        warmup_steps=500,  # The number of warm-up steps
        weight_decay=0.01,  # The weight decay
        logging_dir="./logs",  # The directory for the logs
        save_strategy="epoch",  # Save the model after each epoch
        save_total_limit=3,  # Limit the total number of saved models
    )

    # Define the trainer
    trainer = Trainer(
        model=model,  # The model to be fine-tuned
        args=training_args,  # The training arguments
        train_dataset=dataset["train"],  # The training dataset
        eval_dataset=dataset["validation"],  # The validation dataset
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)


if __name__ == "__main__":

    # Load the YAML configuration file
    with open("conf/local.yml", "r") as yamlfile:
        config = yaml.safe_load(yamlfile)  # Use safe_load to load the YAML configuration file safely
        
    # Load the training data
    dataset_path = config["training_params"]["training_dataset"]
    dataset = load_from_disk(dataset_path)

    # Train the model
    model_path = config["training_params"]["model_path"]
    train_model(dataset, "deepset/roberta-base-squad2", model_path)