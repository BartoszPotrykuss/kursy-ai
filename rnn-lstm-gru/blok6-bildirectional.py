import torch
import torch.nn as nn

layer = nn.LSTM(input_size=3, hidden_size=5, batch_first=True, bidirectional=True)
x = torch.randn(2, 4, 3)
output, (h_n, c_n) = layer(x)
#zgaduje, że zmieni się kształt output bo teraz go dostaniemy z obu stron
print(output.shape)
print(h_n.shape)
print(c_n.shape)