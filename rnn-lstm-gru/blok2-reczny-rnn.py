import torch

input_size = 3
hidden_size = 5
seq_len = 4
x = torch.randn(seq_len, input_size)
W_xh = torch.randn(hidden_size, input_size)
W_hh = torch.randn(hidden_size, hidden_size)
b_h = torch.randn(hidden_size,)
h_t = torch.zeros(hidden_size,)

hidden_states = []

for t in range(seq_len):
    h_t = torch.tanh(W_xh @ x[t] + W_hh @ h_t + b_h)
    print(h_t.shape)
    hidden_states.append(h_t)

print(hidden_states)