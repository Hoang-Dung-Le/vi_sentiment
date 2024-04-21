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

def parse_option():
    parser = argparse.ArgumentParser('Vi Sentiment', add_help=False)

    parser.add_argument('--batch-size', type=int, help="batch size for single GPU", default=32)
    # parser.add_argument('--train_file', type=str, help='path to train dataset', default='./train.csv')
    # parser.add_argument('--val_file', type=str, help='path to val dataset', default='./val.csv')
    # parser.add_argument('--test_file', type=str, help='path to test dataset', default='./test.csv')
    parser.add_argument('--model', type=str, help='path to model')

    args, unparsed = parser.parse_known_args()

    return args

def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    predictions = []
    labels = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        eval_loss += loss.item()
        predictions.extend(outputs.logits.argmax(-1).to('cpu').numpy())
        labels.extend(batch['labels'].to('cpu').numpy())
    eval_acc = np.sum(predictions == labels) / len(labels)

    eval_f1 = f1_score(labels, predictions, average='macro')
    print("-------Result---------")
    print(f"acc: {eval_acc: .4f}")
    print(f"f1: {eval_f1: .4f}")


def main(args):

    phobert = AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    # load data
    print("Loading data...")
    # phobert.load_state_dict(torch.load(args.model))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    phobert.to(device)
    train_dataloader, eval_dataloader = get_dataset(args, tokenizer)
    print("-------eval training data---------")
    evaluate(phobert, tokenizer, train_dataloader, device)
    print("-------eval testing data -------")
    evaluate(phobert, tokenizer, eval_dataloader, device)
    print("------END-------")

if __name__ == '__main__':
    args = parse_option()
    main(args)

