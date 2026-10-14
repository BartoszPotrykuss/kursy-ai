import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


model = LinearModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = 3 * x_data + 2
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

"""
for i in range(1000):
    y_pred = model(x_data)
    loss = loss_fn(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(loss.item())

print(model.linear.weight.item(), model.linear.bias.item())
"""

#zadanie 6
x_data = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_data = x_data ** 2

modelMLP = MLP()
optimizer = torch.optim.Adam(modelMLP.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for i in range(500):
    y_pred = modelMLP(x_data)
    loss = loss_fn(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(loss.item())


#zadanie 7

# 3 klasy, każda skupiona wokół innej wartości x
x_data = torch.cat([
    torch.randn(20, 1) - 3,   # klasa 0, wokół -3
    torch.randn(20, 1),       # klasa 1, wokół 0
    torch.randn(20, 1) + 3,   # klasa 2, wokół +3
])
y_data = torch.cat([
    torch.zeros(20, dtype=torch.long),
    torch.ones(20, dtype=torch.long),
    torch.full((20,), 2, dtype=torch.long)
])


class ModelCEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


modelCEL = ModelCEL()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelCEL.parameters(), lr=0.01)

for i in range(300):
    y_pred = modelCEL(x_data)
    loss = loss_fn(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 30 == 0:
        print(loss.item())

print()
print(x_data[0])
print()
logits = modelCEL(x_data[0:1])
probs = torch.softmax(logits, dim=1)
print(logits)
print(probs)