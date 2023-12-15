import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from SentimentModel import SentimentModel
import random


tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


def tokenize_and_transform(sentence, vocab, tokenizer):
    tokens = tokenizer(sentence)
    token_ids = [vocab[token] for token in tokens]
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


train_data = list(IMDB(split='train'))
random.shuffle(train_data)

vocab = build_vocab_from_iterator(yield_tokens(data_iter for label, data_iter in train_data), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

d_model = 512
num_heads = 8
max_seq_len = 5000
d_ff = 2048
learning_rate = 0.001
num_classes = 2
vocab_size = len(vocab)
num_epochs = 100

model = SentimentModel(d_model, d_ff, num_heads, max_seq_len, num_classes, vocab_size)
model.load_state_dict(torch.load('sentiment_model_state.pth', map_location=torch.device('cpu')))

sentence = ("My visit to the cinema was an underwhelming experience. "
            "I honestly expected more from this director")
processed_sentence = tokenize_and_transform(sentence, vocab, tokenizer)

with torch.no_grad():
    prediction = model(processed_sentence)
    prediction = torch.sigmoid(prediction)
    sentiment = "pozytywne" if prediction.item() > 0.5 else "negatywne"

print(f"Zdanie '{sentence}' jest ocenione jako: {sentiment}")
