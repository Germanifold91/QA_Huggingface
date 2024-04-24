from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering
import json
from datasets import Dataset


def load_and_format_data(data_file, save_dir):
    # Load the data from a JSON file
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Prepare the data in the format required for fine-tuning
    formatted_data = {
        'title': [],
        'context': [],
        'question': [],
        'answers': [],
    }
    for document in data['documents']:
        title = document['title']
        for qa in document['questions_answers']:
            context = document['contexts'][qa['context_index']]
            formatted_data['title'].append(title)
            formatted_data['context'].append(context)
            formatted_data['question'].append(qa['question'])
            formatted_data['answers'].append({
                'text': [qa['answer']],
                'answer_start': [context.find(qa['answer'])],
            })

    # Convert the data to a Hugging Face Dataset
    dataset = Dataset.from_dict(formatted_data)

    # Save the dataset to a directory
    dataset.save_to_disk(save_dir)

    return dataset


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

# Use the functions
dataset = load_and_format_data('/Users/germanifold/Desktop/Germanifold/Python_Projects/loka_qa/data/01_raw/training_objects/training_contexts.json',
                               '/Users/germanifold/Desktop/Germanifold/Python_Projects/loka_qa/data/02_intermediate/training_dataset')

print(dataset['title'][3])
print(dataset['context'][3])
print(dataset['question'][3])
print(dataset['answers'][3])
#train_model(dataset, "deepset/roberta-base-squad2")