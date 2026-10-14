import torch
import torch.nn as nn

layer = nn.LSTM(input_size=3, hidden_size=5, batch_first=True)

x = torch.randn(2, 4, 3)

output, (h_n, c_n) = layer(x)

print(output)
print(h_n, c_n)