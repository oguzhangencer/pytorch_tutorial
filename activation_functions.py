import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)

output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)

output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)

output = torch.relu(x)
print(output)
relu = nn.ReLU()
output = relu(x)
print(output)

output = F.leaky_relu(x)
print(output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)
