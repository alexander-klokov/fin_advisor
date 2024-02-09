import fire
import logging

import numpy as np
import evaluate

from datasets import load_dataset

from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments

from ..utils.log_memory import log_gpu_memory, log_ram

metric = evaluate.load("accuracy")

def get_data():
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    return train_dataset, eval_dataset

# def train(data: str, output: str):
def train():
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("#" * 64)

    log_gpu_memory()
    log_ram()

    logger.info("#" * 64)

    train_dataset, eval_dataset = get_data()

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    training_args = TrainingArguments(output_dir="test_trainer")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()


if __name__ == "__main__":
    fire.Fire(train)
