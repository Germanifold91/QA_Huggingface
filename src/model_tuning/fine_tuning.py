from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering


def train_model(dataset, model_name):
    # Load the model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # The output directory
        num_train_epochs=3,  # The number of training epochs
        per_device_train_batch_size=16,  # The batch size for training
        per_device_eval_batch_size=64,  # The batch size for evaluation
        warmup_steps=500,  # The number of warm-up steps
        weight_decay=0.01,  # The weight decay
        logging_dir="./logs",  # The directory for the logs
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
