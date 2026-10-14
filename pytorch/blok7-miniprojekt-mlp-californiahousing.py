import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModelMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


housing = fetch_california_housing(as_frame=True)

df = housing.frame

features = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
]

scaler = StandardScaler()
X = df[features]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.15/(1-0.15))

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_val = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

train_data = MyDataset(X_train_scaled, y_train)
val_data = MyDataset(X_val_scaled, y_val)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

model = ModelMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


for epoch in range(50):
    model.train()
    train_loss_sum = 0.0

    for batch_X, batch_y in train_loader:
        y_pred = model(batch_X)
        loss = loss_fn(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()

    train_loss_avg = train_loss_sum / len(train_loader)

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            val_loss_sum += loss.item()

    val_loss_avg = val_loss_sum / len(val_loader)

    if epoch % 5 == 0:
        print(f"{epoch}: train_loss={train_loss_avg:.4f}, val_loss={val_loss_avg:.4f}")


model.eval()
with torch.no_grad():
    y_pred = model(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")