import torch

input_size = 1
hidden_size = 10
seq_len = 50
x = torch.randn(seq_len, input_size)
W_xh = torch.randn(hidden_size, input_size, requires_grad=True)
W_hh = torch.randn(hidden_size, hidden_size, requires_grad=True) * 0.1
b_h = torch.randn(hidden_size, requires_grad=True)
h_t = torch.zeros(hidden_size,)

hidden_states = []

for t in range(seq_len):
    h_t = torch.tanh(W_xh @ x[t] + W_hh @ h_t + b_h)
    h_t.retain_grad()
    hidden_states.append(h_t)

#print(hidden_states)

loss = (hidden_states[-1] ** 2).sum()
loss.backward()

for h_s in hidden_states:
    print(h_s.grad.norm())