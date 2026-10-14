import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = 3 * x_data + 2


dataset = MyDataset(x_data, y_data)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_x, batch_y in loader:
    print(batch_x, batch_y)
