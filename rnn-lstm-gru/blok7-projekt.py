import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
df["Month"] = pd.to_datetime(df["Month"])

print(df.shape)
print(df.info())
print(df.head())

"""
plt.plot(df["Month"], df["Passengers"], )
plt.xlabel("Miesiąc")
plt.ylabel("Pasażerowie")
plt.show()
"""


def create_sequences(data, window_size):
    X = np.zeros((len(data) - window_size, window_size))
    y = np.empty((len(data) - window_size,))
    for i in range(len(data) - window_size):
        X[i] = data[i:i+window_size, 0]      # kolumna 0, bo scaler zwraca (n, 1)
        y[i] = data[i+window_size, 0]
    return X, y


df["Passengers"] = np.log(df["Passengers"])
df_train = df[:-24]
df_test = df[-24:]

scaler = MinMaxScaler()
df_train_scaled = scaler.fit_transform(df_train[["Passengers"]])
df_test_scaled = scaler.transform(df_test[["Passengers"]])

df_test_scaled_extended = np.concatenate([df_train_scaled[-12:], df_test_scaled])

X_train, y_train = create_sequences(df_train_scaled, window_size=12)
X_test, y_test = create_sequences(df_test_scaled_extended, window_size=12)
#print(X_test)


class PassengersDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(-1)
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] 


train_data = PassengersDataset(X_train, y_train)
test_data = PassengersDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


class PassengerLSTM(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size=hidden_size, batch_first=True, input_size=1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)


model = PassengerLSTM()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    model.train()
    epoch_losses = []
    for batch_X, batch_y in train_loader:
        y_pred = model(batch_X)
        loss = loss_fn(y_pred, batch_y.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y.unsqueeze(-1))
            test_losses.append(loss.item())

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch},"
            f" avg train loss: {sum(epoch_losses)/len(epoch_losses):.4f}, "
            f"avg test loss: {sum(test_losses)/len(test_losses):.4f}"
        )

model.eval()
predictions = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        y_pred = model(batch_X)
        predictions.append(y_pred)

predictions = torch.cat(predictions).numpy()
predictions_scaled = np.exp(scaler.inverse_transform(predictions))

y_test_reshaped = y_test.reshape(-1, 1)
y_test_scaled = np.exp(scaler.inverse_transform(y_test_reshaped))

plt.plot(y_test_scaled, label="Rzeczywiste")
plt.plot(predictions_scaled, label="Predykcja")
plt.legend()
plt.xlabel("Miesiąc (test)")
plt.ylabel("Pasażerowie")
plt.show()

print(np.mean(np.abs(predictions_scaled - y_test_scaled)))


df["Passengers"] = np.exp(df["Passengers"])
predictions_naive = []
for i in range(24):
    pred = df.iloc[-24+i-12, 1]
    predictions_naive.append(pred)

predictions_naive = np.array(predictions_naive).reshape(-1, 1)
print(np.mean(np.abs(predictions_naive - y_test_scaled)))
