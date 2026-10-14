import torch
import torch.nn as nn

layer = nn.GRU(input_size=3, hidden_size=5, batch_first=True)
x = torch.randn(2, 4, 3)

for element in layer(x):
    print(element.shape)