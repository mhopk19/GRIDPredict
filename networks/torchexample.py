import torch
import math
import matplotlib.pyplot as plt
from torch.utils import data
import torch.nn as nn
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import time 
import sys

print("Current Python version: ", sys.version)
print("Current Pytorch version: ", torch.__version__)

### Creating the dataset
 
def synthetic_data(m, c, num_examples):
    # Generate y = mX + bias(c) + noise
    X = torch.normal(0, 1, (num_examples, len(m)))
    y = torch.matmul(X, m) + c
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_m = torch.tensor([2, -3.4])
true_c = 4.2
total_data_points = 1000
features, labels = synthetic_data(true_m, true_c, total_data_points)

#Step 3: Read dataset and create small batch
def load_array(data_arrays, batch_size, is_train=True):  
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# defining batch size and creating data loader object
batch_size = 50
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# Define model & initialize

# Create single layer feed-forward network with 2 inputs and 1 outputs.
net = nn.Linear(2, 1)

# example of loading models
load_params = True
model_filepath = "model.pth"

# if the load_params argument is set and the model_filepath exists load previous model
if load_params and os.path.exists(model_filepath):
    net = torch.load(model_filepath)
    print("loaded the model from {}".format(model_filepath))
else:
    print("chosen file is not a valid file path")
    
# loading and saving models is cruical when training takes very long
# we can save the model at different checkpoints and resume training later

#Initialize model params 
net.weight.data.normal_(0, 0.01)
net.bias.data.fill_(0)

# mean squared error loss function
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 5
# indicate to the model that we are training
net.train()
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        # sets gradients to zero 
        # we do not want to be using old gradients in current batch
        trainer.zero_grad() 
        # perform backpropagation
        l.backward() # back propagation
        # update the model parameters based on the training paradigm defined by trainer
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    
# indicate to the model that we are evaluating
net.eval()

# Comparing learned results with the actual parameters used to generate data
m = net.weight.data
print('error in estimating m:', true_m - m.reshape(true_m.shape))
c = net.bias.data
print('error in estimating c:', true_c - c)


