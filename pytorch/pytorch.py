import torch

#zadanie 1
x = torch.tensor(3.0, requires_grad=True)
y1 = x ** 2
y1.backward()
print(x.grad)

x.grad.zero_()

y2 = x ** 3
y2.backward()
print(x.grad)

#zadanie 2
x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1
for i in range(20):
    y = (x - 5) ** 2
    y.backward()
    with torch.no_grad(): x -= learning_rate * x.grad
    x.grad.zero_()
    print(x.item(), y.item())

#zadanie 3
print()
print("Linear regression przez gradient descent")
learning_rate = 0.01
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = 3 * x_data + 2

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

for i in range(1000):
    y_pred = w * x_data + b
    loss = ((y_pred - y_data)**2).mean()
    loss.backward()
    with torch.no_grad(): w -= learning_rate * w.grad; b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if i % 100 == 0:
        print(w.item(), b.item())