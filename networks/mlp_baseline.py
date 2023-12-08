import torch
import math
import matplotlib.pyplot as plt
from torch.utils import data
import torch.nn as nn
import os
import re
import sys
#sys.path.append(".")
sys.path.append("./../")
from classes.data_loader import *

def kl_sum_loss(output, target):
    def univariate_gaussian_kl(mu1,mu2,sigma1,sigma2):
        term1 = torch.log(sigma2**2/sigma1**2)
        term2 = (sigma1**2+(mu1-mu2)**2)/(2*sigma2**2)
        term3 = -1/2
        #print("term1", term1)
        #print("term2", term2)
        #print("kl", term1+term2+term3)
        return term1+term2+term3
    measurement_error = 0.001
    loss = 0
    #print("output.shape", output.shape)
    #print("output sample", output[5][0])
    #print("target.shape", target.shape)
    for i in range(output.shape[1]):
        loss = loss + univariate_gaussian_kl(output[:,i][0], target[i],
                                          output[:,i][1], measurement_error)
    return loss

class MLP_baseline(nn.Module):
    def __init__(self, num_inputs, num_features):
        super(MLP_baseline, self).__init__()
        self.layer1_nodes = 256
        self.layer2_nodes = 64
        self.output_layers = 1
        self.feature_size = num_inputs * num_features
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size, self.layer1_nodes),
            nn.ReLU(),
            nn.Linear(self.layer1_nodes, self.layer2_nodes),
            nn.ReLU(),
            nn.Linear(self.layer2_nodes, self.output_layers)
        )

    def forward(self,x):
        return self.layers(x.view(-1, self.feature_size))
    
    def load_model(self,file):
        self.load_state_dict(torch.load(file))

    def save_model(self,file):
        torch.save(self.state_dict(), file)

def train_mlp_model(mlp_model, data_loader, epochs = 1, save_file = ""):
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
    loss_fn = kl_sum_loss
    print("Beginning Training...")
    num_epochs = 1
    for n in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            if (batch_idx % 100 == 0):
                print("batch {}".format(batch_idx))
            x, y = next(iter(data_loader))
            y_pred = mlp_model(x)
            loss = loss_fn(y_pred, y)
            total_loss = loss + total_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx > 100):
                break
        print("epoch loss:", loss.item())
    if (save_file != ""):
        torch.save(mlp_model, save_file)

    return mlp_model

def get_mlp_output(input, mlp_model):
    y_pred = mlp_model(input).ravel()
    formatted_output = y_pred
    return formatted_output

if __name__ == "__main__":
    os.chdir("./..")
    df = pd.read_csv("cleaned_HomeC.csv", low_memory = False)
    train_dataset = smarthome_dataset("cleaned_HomeC.csv" , dayahead = True,
                                                    start_index = 10,
                                                    end_index = 4000)
    test_dataset = smarthome_dataset("cleaned_HomeC.csv" , dayahead = True,
                                                    start_index = 25000,
                                                    end_index = 30000)
    
    batch_size = 50

    train_dataloader = DataLoader(train_dataset, batch_size = 50,
                            shuffle = True, num_workers = 0)
    test_dataloader = DataLoader(test_dataset, batch_size = 50,
                            shuffle = True, num_workers = 0)
    
    mlp_model = MLP_baseline(1,9)

    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

    loss_fn = kl_sum_loss

    print("Beginning Training...")
    num_epochs = 1
    for n in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            print("batch {}".format(batch_idx))
            x, y = next(iter(train_dataloader))
            y_pred = mlp_model(x)
            print(y_pred.shape)
            loss = loss_fn(y_pred, y)
            total_loss = loss + total_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch loss:", loss.item())
    
    mlp_model.save_model("model.pth")
    mlp_model.load_model("model.pth")

    
