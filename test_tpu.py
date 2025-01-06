import torch

import torch_xla
import torch_xla.core.xla_model as xm

# Creates a random tensor on xla:1 (a Cloud TPU core)
dev = xm.xla_device()
t1 = torch.ones(3, 3, device = dev)
print(t1)

# Creating a tensor on the second Cloud TPU core
second_dev = xm.xla_device(n=2, devkind='TPU')
t2 = torch.zeros(3, 3, device = second_dev)
print(t2)

a = torch.randn(2, 2, device = dev)
b = torch.randn(2, 2, device = dev)
print(a + b)
print(b * 2)
print(torch.matmul(a, b))

# Creates random filters and inputs to a 1D convolution
filters = torch.randn(33, 16, 3, device = dev)
inputs = torch.randn(20, 16, 50, device = dev)
torch.nn.functional.conv1d(inputs, filters)

# Creates a tensor on the CPU (device='cpu' is unnecessary and only added for clarity)
t_cpu = torch.randn(2, 2, device='cpu')
print(t_cpu)

t_tpu = t_cpu.to(dev)
print(t_tpu)

t_cpu_again = t_tpu.to('cpu')
print(t_cpu_again)

import torch.nn as nn
import torch.nn.functional as F

# Simple example network from 
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Places network on the default TPU core
net = Net().to(dev)

# Creates random input on the default TPU core
input = torch.randn(1, 1, 32, 32, device=dev)

# Runs network
out = net(input)
print(out)