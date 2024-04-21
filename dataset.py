from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import torch
from sklearn.preprocessing import LabelEncoder


def get_dataset(args, tokenizer):
    dataset = load_dataset('uitnlp/vietnamese_students_feedback')
    
    dataset = dataset.remove_columns('topic')
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("sentiment", "labels")


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    test_dataset = tokenized_datasets["test"].shuffle(seed=42)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    return train_dataloader, test_dataloader