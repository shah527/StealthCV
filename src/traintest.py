'''
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

# Sample dataset
dataset = [
    ("Apple is a Fortune 50 company.", 1),
    ("Walmart is also on the list.", 0),
    ("Microsoft and Google are well-known companies.", 0),
    ("I love using Apple products.", 1)
]

train_dataset = dataset[:3]
valid_dataset = dataset[3:]

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=32)
        tokens['label'] = label
        return tokens

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = TextDataset(train_dataset, tokenizer)
valid_dataset = TextDataset(valid_dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2)
valid_loader = DataLoader(valid_dataset, batch_size=2)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        labels = torch.tensor(batch['label'])
        inputs = {key: torch.tensor(val) for key, val in batch.items() if key != 'label'}
        outputs = model(**inputs)
        loss = loss_function(outputs.logits, labels)
        loss.backward()
        optimizer.step()

# Simple Evaluation on Validation Set
model.eval()
for batch in valid_loader:
    with torch.no_grad():
        inputs = {key: torch.tensor(val) for key, val in batch.items() if key != 'label'}
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        print(predictions) # Output prediction
'''