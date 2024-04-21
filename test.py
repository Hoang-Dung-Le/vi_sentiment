from transformers import AutoModelForSequenceClassification, AutoTokenizer

phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base",num_labels=3)