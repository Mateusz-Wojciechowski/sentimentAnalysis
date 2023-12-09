import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from SentimentModel import SentimentModel
import torch.optim as optim
import torch.nn as nn

# Tokenizer
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def process_text(text, vocab):
    return torch.tensor(vocab(tokenizer(text)))


def collate_batch(batch, vocab):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(torch.tensor(1 if label == 'pos' else 0))
        processed_text = process_text(text, vocab)
        text_list.append(processed_text)
    return torch.stack(label_list), pad_sequence(text_list, padding_value=vocab["<pad>"], batch_first=True)


train_iter, test_iter = IMDB()
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

collate_fn = partial(collate_batch, vocab=vocab)
train_loader = DataLoader(train_iter, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_iter, batch_size=32, shuffle=False, collate_fn=collate_fn)

d_model = 512
num_heads = 8
max_seq_len = 5000
d_ff = 2048
learning_rate = 0.001
num_classes = 3
vocab_size = len(vocab)
num_epochs = 100


model = SentimentModel(d_model, d_ff, num_heads, max_seq_len, num_classes, vocab_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        print(f"batch")
        labels, sequences = batch

        output = model(sequences)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Loss in epoch {epoch} is {total_loss}")

