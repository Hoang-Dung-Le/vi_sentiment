from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import torch
from sklearn.preprocessing import LabelEncoder

# def get_dataset(args, tokenizer):
#     dataset = load_dataset("csv", data_files={"train": args.train_file,
#                                           "val": args.val_file,
#                                           "test": args.test_file})
    
#     dataset = dataset.remove_columns(["rate", "Unnamed: 3"])


#     def tokenize_function(examples):
#         encoded_inputs = tokenizer(examples["comment"], padding="max_length", truncation=True, return_tensors='pt')

#         labels = examples["label"]
#         label_encoder = LabelEncoder()
#         encoded_labels = label_encoder.fit_transform(labels)
#         encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

#         encoded_example = {**encoded_inputs, "label": encoded_labels}

#         return encoded_example
    
#     tokenized_datasets = dataset.map(tokenize_function, batched=True)

#     tokenized_datasets = tokenized_datasets.remove_columns(["comment"])
#     tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#     tokenized_datasets.set_format("torch")

#     train_dataset = tokenized_datasets["train"].shuffle(seed=42)
#     eval_dataset = tokenized_datasets["val"].shuffle(seed=42)
#     test_dataset = tokenized_datasets["test"].shuffle(seed=42)

#     train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
#     eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
#     return train_dataloader, eval_dataloader, test_dataloader


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