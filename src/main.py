from torch import Tensor
from torch.nn import Module
# Layers
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
# Activations
from torch.nn import LogSoftmax
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import ELU
from torch.nn import LeakyReLU
# Loss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

import torch
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

flatten = Flatten()

class Network(Module):
    def __init__(self,channels_num):
        super(Network,self).__init__()

        self.relu = ReLU()
        self.elu = ELU()
        self.sigmoid = Sigmoid()
        self.logsoftmax = LogSoftmax()
        self.leakyrelu = LeakyReLU()

        self.conv1 = Conv2d(in_channels=channels_num,out_channels=20,kernel_size=(3,3))
        self.maxpool1 = MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv2 = Conv2d(in_channels=20,out_channels=50,kernel_size=(3,3))
        self.maxpool2 = MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.fc1 = Linear(in_features=167700,out_features=50)
        self.fc2 = Linear(in_features=50,out_features=2)

        #self.LogSoftmax = LogSoftmax(dim=1)

    def forward(self,x):
        x = self.conv1(x)
        #x = self.sigmoid(x)
        x = self.elu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        #x = self.sigmoid(x)
        x = self.elu(x)
        x = self.maxpool2(x)

        x = flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)

        x = self.fc2(x)
        x = self.logsoftmax(x)
        #x = self.LogSoftmax(x)

        return x

nn = Network(3)

criterion = MSELoss()#CrossEntropyLoss()

optimizer = optim.SGD(nn.parameters(), lr=0.003, momentum=0.0)

image_path = "data/1.png"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((320,180)),
    transforms.ToTensor(),
])
input_image = transform(image).unsqueeze(0)
target = Tensor([[.1,.5]])

for i in range(200):
    outputs = nn(input_image)

    print(outputs)

    loss = criterion(outputs,target)

    loss.backward()

    optimizer.step()
    
    #print("Loss: ",loss.item())

nn.eval()

with torch.no_grad():
    output = nn(input_image).tolist()[0]

print(output)
