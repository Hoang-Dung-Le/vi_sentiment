from datasets import load_dataset
import argparse
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

def parse_option():
    parser = argparse.ArgumentParser('Vi Sentiment', add_help=False)

    parser.add_argument('--batch-size', type=int, help="batch size for single GPU", default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_file', type=str, help='path to train dataset', default='./train.csv')
    parser.add_argument('--val_file', type=str, help='path to val dataset', default='./val.csv')
    parser.add_argument('--test_file', type=str, help='path to test dataset', default='./test.csv')
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)

    args, unparsed = parser.parse_known_args()

    return args



def main(args):
    dataset = load_dataset("csv", data_files={"train": args.train_file,
                                          "val": args.val_file,
                                          "test": args.test_file})
    
    dataset = dataset.remove_columns(["rate", "Unnamed: 3"])

    phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base",num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    def tokenize_function(examples):
        encoded_inputs = tokenizer(examples["comment"], padding="max_length", truncation=True, return_tensors='pt')

        labels = examples["label"]
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

        encoded_example = {**encoded_inputs, "label": encoded_labels}

        return encoded_example
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["comment"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["val"].shuffle(seed=42)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    num_epochs = args.epochs
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(phobert.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    phobert.to(device)

    progress_bar = tqdm(range(num_training_steps))

    train_losses = []
    eval_losses = []
    eval_accs = []
    eval_f1s = []

    phobert.train()
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = phobert(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            epoch_train_loss += loss.item()  # accumulate loss for each batch

            predictions = outputs.logits.argmax(-1).to('cpu')
            label_ids = batch['labels'].to('cpu')

            progress_bar.update(1)

        train_losses.append(epoch_train_loss)

        phobert.eval()
        eval_loss = 0
        eval_num_correct = 0
        eval_num_examples = 0
        predictions = []
        labels = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = phobert(**batch)

            logits = outputs.logits
            loss = outputs.loss
            eval_loss += loss.item()
            predictions.extend(outputs.logits.argmax(-1).to('cpu').numpy())
            labels.extend(batch['labels'].to('cpu').numpy())
        eval_acc = np.sum(predictions == labels) / len(labels)

        eval_f1 = f1_score(labels, predictions, average='macro')

        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        eval_f1s.append(eval_f1)

        print(f"Epoch: {epoch+1}, Training Loss: {epoch_train_loss:.4f},  Validation loss {eval_loss: .4f},
               F1 Score: {eval_f1}")
        
    now = datetime.now()
    torch.save({
        'model_state_dict': phobert.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,  # Add current epoch
    }, f'model_{now}.pt')

    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Validation Loss')
    plt.plot(eval_f1s, label='Validation F1')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('./training.png')

