import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from SentimentModel import SentimentModel
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Tokenizer
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

def process_text(text, vocab):
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.long).to(device)

def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_tensor = torch.tensor([label-1], dtype=torch.float).to(device)
        processed_text = process_text(text, vocab)
        label_list.append(label_tensor)
        text_list.append(processed_text)
    return torch.stack(label_list).to(device), pad_sequence(text_list, padding_value=vocab["<pad>"], batch_first=True).to(device)

def calculate_accuracy(preds, y):
    preds = torch.sigmoid(preds)
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

train_data = list(IMDB(split='train'))
random.shuffle(train_data)

vocab = build_vocab_from_iterator(yield_tokens(data_iter for label, data_iter in train_data), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

train_loader = DataLoader(train_data, batch_size=32, shuffle=False, collate_fn=collate_batch)

test_data = list(IMDB(split='test'))
random.shuffle(test_data)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_batch)

d_model = 512
num_heads = 8
max_seq_len = 5000
d_ff = 2048
learning_rate = 0.001
num_classes = 2
vocab_size = len(vocab)
num_epochs = 100

model = SentimentModel(d_model, d_ff, num_heads, max_seq_len, num_classes, vocab_size)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    total_loss = 0
    total_accuracy = 0
    total_examples = 0

    model.train()
    for labels, sequences in train_loader:
        output = model(sequences)
        loss = loss_fn(output, labels)
        accuracy = calculate_accuracy(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()
        total_examples += labels.size(0)

    print(f"Loss in epoch {epoch + 1} is {total_loss}")
    print(f"Accuracy in epoch {epoch + 1} is {total_accuracy / total_examples}")
