import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import data_processor
import model
import train
import test
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from behaviorPrediction.model import TabularNN

x_data, y_data = data_processor.prepare_data('train.csv')

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

# Преобразуем данные в тензоры
X_train_tensor = X_train.detach().clone().float()
y_train_tensor = y_train.detach().clone().float().unsqueeze(1)
X_test_tensor = X_test.detach().clone().float()
y_test_tensor = y_test.detach().clone().float().unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используем {device} устройство.")

num_features = X_train_tensor.shape[1]

model = TabularNN(num_features).to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train.train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test.test_loop(test_dataloader, model, loss_fn,device)

print("Done!")




