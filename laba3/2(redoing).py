'''Задание
Обучите модель нейросети, выводя промежуточные данные: номер эпохи, точность на тренировочных данных, точность на тестовых данных. 
Примечание: лучшие результаты сохраняйте в файл.
В LSTM-слое установите параметр lstm_layers равным 4. Как изменился результат обучения в плане точности и скорости?'''

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv("reviews_preprocessed.csv")

all_words = " ".join(data.processed.values).split()
counter = Counter(all_words)

vocabulary = sorted(counter, key=counter.get, reverse=True)

int2word = dict(enumerate(vocabulary, 1))
int2word[0] = "PAD"

word2int = {word: id for id, word in int2word.items()}

reviews = data.processed.values
all_words = " ".join(reviews).split()

review_enc = [[word2int[word] for word in review.split()] for review in reviews]

sequence_length = 256
reviews_padding = np.full((len(review_enc), sequence_length), word2int['PAD'], dtype=int)

for i, row in enumerate(review_enc):
    reviews_padding[i, :len(row)] = np.array(row)[:sequence_length]

labels = data.label.to_numpy()

train_len = 0.6
val_len = 0.2
train_last_index = int(len(reviews_padding) * train_len)
val_last_index = int(len(reviews_padding) * (train_len + val_len))

train_x = reviews_padding[:train_last_index]
train_y = labels[:train_last_index]

val_x = reviews_padding[train_last_index:val_last_index]
val_y = labels[train_last_index:val_last_index]

test_x = reviews_padding[val_last_index:]
test_y = labels[val_last_index:]

print(f"Train: {len(train_x)}, Validation: {len(val_x)}, Test: {len(test_x)}")

train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 128
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.3):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        hidden_last = hidden[-1, :, :]
        
        dropped = self.dropout(hidden_last)
        output = self.fc(dropped)
        predictions = self.sigmoid(output)
        return predictions


def create_model(vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=4, dropout=0.3):
    model = TextModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
    return model


vocab_size = len(word2int)
model = create_model(vocab_size, n_layers=4)  

print(f"Количество параметров модели: {sum(p.numel() for p in model.parameters()):,}")
print(f"Количество слоев LSTM: 4")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def accuracy(predictions, labels):
    rounded_preds = torch.round(predictions)
    correct = (rounded_preds == labels).float()
    return correct.sum() / len(correct)


model_path = './best_model_lstm4.pth'
best_val_loss = float('inf')
best_val_acc = 0

start_time = time.time()

for epoch in range(5):
    model.train()
    train_loss = 0
    train_acc = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.long()
        batch_y = batch_y.float().unsqueeze(1)
        
        lengths = (batch_x != word2int['PAD']).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)
        
        optimizer.zero_grad()
        predictions = model(batch_x, lengths)
        loss = criterion(predictions, batch_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        acc = accuracy(predictions, batch_y)
        train_loss += loss.item()
        train_acc += acc.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    
    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.long()
            batch_y = batch_y.float().unsqueeze(1)
            
            lengths = (batch_x != word2int['PAD']).sum(dim=1)
            lengths = torch.clamp(lengths, min=1)
            
            predictions = model(batch_x, lengths)
            loss = criterion(predictions, batch_y)
            acc = accuracy(predictions, batch_y)
            
            val_loss += loss.item()
            val_acc += acc.item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_acc': avg_val_acc,
            'val_loss': avg_val_loss
        }, model_path)
        print(f'Эпоха {epoch+1}. Сохранена лучшая модель (Val Acc: {avg_val_acc:.4f}, Loss: {avg_val_loss:.4f})')
    
    print(f'Эпоха {epoch+1}: Train Acc = {avg_train_acc:.4f}, Train Loss = {avg_train_loss:.4f} | Val Acc = {avg_val_acc:.4f}, Val Loss = {avg_val_loss:.4f}')

training_time = time.time() - start_time

print("\n" + "="*50)
print("ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ")
print("="*50)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Загружена модель с эпохи {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")

model.eval()
test_loss = 0
test_acc = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.long()
        batch_y = batch_y.float().unsqueeze(1)
        
        lengths = (batch_x != word2int['PAD']).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)
        
        predictions = model(batch_x, lengths)
        loss = criterion(predictions, batch_y)
        acc = accuracy(predictions, batch_y)
        
        test_loss += loss.item()
        test_acc += acc.item()

avg_test_loss = test_loss / len(test_loader)
avg_test_acc = test_acc / len(test_loader)

print(f"Результаты на тестовой выборке:")
print(f"Test Accuracy = {avg_test_acc:.4f}")
print(f" Test Loss = {avg_test_loss:.4f}")
print(f"\nОбщее время обучения: {training_time:.2f} секунд")
