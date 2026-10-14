import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
X = torch.rand(500, 5)
true_w = torch.tensor([1.5, -2.0, 0.5, 3.0, -1.0])
y = X @ true_w + 0.1 * torch.randn(500)
y = y.reshape(-1, 1)

X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_fn = nn.MSELoss()

best_val_loss = float("inf")
patience = 5
epochs_without_improvement = 0

for epoch in range(100):

    model.train()
    train_loss_sum = 0.0

    for batch_x, batch_y in train_loader:
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()

    avg_train_loss = train_loss_sum / len(train_loader)

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            val_loss_sum += loss.item()

    avg_val_loss = val_loss_sum / len(val_loader)

    if epoch % 10 == 0:
        print(f"{epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping na epoce {epoch}")
            print(f"{epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            break