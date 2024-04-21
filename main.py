import argparse
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import string
from focal_loss import FocalLoss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from dataset import get_dataset
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
    parser.add_argument("--save_model", type=str, help="path to save model")

    args, unparsed = parser.parse_known_args()

    return args



def main(args):

    phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base",num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    for name, layer in phobert.named_modules():
        if name == "classifier":
            continue
        layer.requires_grad = False

    train_dataloader, eval_dataloader = get_dataset(args, tokenizer)

    num_epochs = args.epochs
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(phobert.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    phobert.to(device)

    progress_bar = tqdm(range(num_training_steps))

    criterion = FocalLoss(reduction='mean')

    train_losses = []
    eval_losses = []
    eval_accs = []
    eval_f1s = []

    phobert.train()
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = phobert(**batch)
            # loss = outputs.loss
            logits = outputs.logits
            loss = criterion(logits, batch["labels"])
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            epoch_train_loss += loss.item()  # accumulate loss for each batch

            predictions = outputs.logits.argmax(-1).to('cpu')
            # label_ids = batch['labels'].to('cpu')

            progress_bar.update(1)

        train_losses.append(epoch_train_loss)

        phobert.eval()
        eval_loss = 0
        predictions = []
        labels = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = phobert(**batch)

            loss = outputs.loss
            eval_loss += loss.item()
            predictions.extend(outputs.logits.argmax(-1).to('cpu').numpy())
            labels.extend(batch['labels'].to('cpu').numpy())
        eval_acc = np.sum(predictions == labels) / len(labels)

        eval_f1 = f1_score(labels, predictions, average='macro')

        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        eval_f1s.append(eval_f1)

        print(f"Epoch: {epoch+1}, Training Loss: {epoch_train_loss:.4f},  Validation loss {eval_loss: .4f}, F1 Score: {eval_f1}")
        
    now = datetime.now()
    name_model = args.save_model + "/" + "model_" + str(now)
    torch.save({
        'model_state_dict': phobert.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
    }, name_model)

    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Validation Loss')
    plt.plot(eval_f1s, label='Validation F1')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('./training.png')


if __name__ == '__main__':
    args = parse_option()
    main(args)