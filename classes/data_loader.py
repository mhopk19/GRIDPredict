import numpy as np
import pandas as pd
import copy
import torch
import sys
import os
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from torch.utils.data import Dataset, DataLoader
sys.path.append("./../")
import classes.powerdf as pdf
import matplotlib.pyplot as plt


class smarthome_dataset(Dataset):
    def __init__(self, csv_file, dayahead = False,
                 start_index = 0,
                 end_index = None):
        """
        Arguments:
        csv_file (string): Path to the csv file with annotations.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.df = pd.read_csv(csv_file)
        if (end_index == None):
            end_index = len(self.df) - 1
        self.df = self.df.truncate(before = start_index, after = end_index)
        self.dayahead = dayahead
        self.input_type = "weather"
        self.output_type = "solar"
        self.input_columns = 9

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns input tensors (weather features) and output (solar) over an hour
        dayahead: whether data values for each input are returned over next 24h
        """
        if (self.input_type == "weather"):
            input_columns = ['windSpeed','cloudCover','windBearing','precipIntensity','dewPoint','pressure','apparentTemperature','visibility','humidity']
        if (self.output_type == "solar"):
            output_columns = "gen_Sol"

        xdata = self.df[input_columns] 
        xdata = xdata.iloc[idx]

        if (self.dayahead):
            # getting per-hour weather data at each hour as an independent variable
            for i in range(23):
                new_xdata = self.df[input_columns]
                new_xdata = new_xdata.iloc[idx + (i*60) :idx + (i+1) * 60]
                new_xdata = np.sum(new_xdata, axis = 0)
                xdata = np.vstack((xdata,new_xdata))

            # getting the cumulative solar generation over the next 24 hours
            ydata = self.df[output_columns].iloc[idx+23*60:idx+24*60]
            labels = torch.tensor(np.sum(ydata))
        else:
            ydata = self.df[output_columns].iloc[idx:idx+60]
            labels = torch.tensor(np.sum(ydata))

        inputs = torch.tensor(xdata)
        return inputs.float(), labels.float()


if __name__ == "__main__":
    os.chdir("./..")
    df = pd.read_csv("cleaned_HomeC.csv", low_memory = False)
    dataset = smarthome_dataset("cleaned_HomeC.csv" , dayahead = True)

    print("dataset length", len(dataset))

    # fill missing values in
    df.fillna(value = 0)
    train_df = df.iloc[:3000]
    test_df = df.iloc[4000:4400]
    df = df.iloc[:]

    print("getting data", dataset[1])

    # 24 hours is 1440 time instants on this scale

    dataloader = DataLoader(dataset, batch_size=10,
                            shuffle=True, num_workers=0)

    open = True
    while (open):
        try:
            inputs, label = next(iter(dataloader))  
            print(inputs.shape)
            #print("input {} : label {}".format(inputs,label))
            open = True
        except:
            open = False

