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
sys.path.append("./..")
import classes.powerdf as pdf
import matplotlib.pyplot as plt
from classes.data_loader import smarthome_dataset
from networks.mlp_baseline import *
from networks.bn_with_causality import *
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./../cleaned_HomeC.csv", low_memory = False)
dataset = smarthome_dataset("./../cleaned_HomeC.csv", dayahead = False)
print("dataset length", len(dataset))


df.fillna(value = 0)
train_df = df.iloc[:3000]
test_df = df.iloc[4000:4400]
df = df.iloc[:]

traindata_loader = DataLoader(dataset, batch_size=10, shuffle=True,
                              num_workers=0)
testdata_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                             num_workers=0)

def compare_models(model_output_funcs, model_args, data_loader):
    error_tensor = np.zeros((len(model_output_funcs), len(data_loader)))
    for batch_idx, (x, y) in enumerate(data_loader):
        print("batch {}".format(batch_idx))
        x, y = next(iter(data_loader))
        for i,model in enumerate(model_output_funcs):
            y_pred = model_output_funcs[i](x, *model_args[i])
            error_tensor[i][batch_idx] = torch.square(y - y_pred).item()
        if (batch_idx > 100):
            break

    return error_tensor

def plot_error_statistics(model_names, error_tensor, title):
    model_names = tuple(model_names)
    num_models = len(model_names)
    error_statistics = {"mean":[0]*num_models, "max":[0]*num_models, "min":[0]*num_models}
    for i, model in enumerate(model_names):
        error_statistics["mean"][list(model_names).index(model)] = np.mean(error_tensor[i])
        error_statistics["max"][list(model_names).index(model)] = np.max(error_tensor[i])
        error_statistics["min"][list(model_names).index(model)] = np.min(error_tensor[i])

    x_coords = np.arange(len(model_names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in error_statistics.items():
        offset = width * multiplier
        rects = ax.bar(x_coords + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title(title)
    ax.set_xticks(x_coords + width, model_names)
    ax.legend(loc='upper left', ncols=3)

    plt.show()


# defining Bayesian Network model


file_path = './../cleaned_HomeC.csv'
weather_variables = ['temperature', 'humidity', 'visibility', 'pressure', 'windSpeed', 'cloudCover', 'precipIntensity', 'dewPoint']
appliance_variables = ['Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 'Garage door', 'Barn', 'Well', 'Microwave', 'Living room', 'Furnace', 'Kitchen']
pred_variables = ['gen_Sol']
#network_edges = weather_variables + appliance_variables + pred_variables
network_edges = run_pipeline(file_path, weather_variables,
                appliance_variables,
                pred_variables)
network_edges = utils.get_component_containing(nx.DiGraph(network_edges), "gen_Sol")
solver, evidence_nodes = train_bn_model(network_edges, train_df)
# defining MLP model

mlp_model = MLP_baseline(1,9)
mlp_model = train_mlp_model(mlp_model, traindata_loader)

# comparing models
print("evidence_nodes", evidence_nodes)
model_funcs = [get_bn_output, get_mlp_output]
model_args = [[evidence_nodes, solver],[mlp_model]]
error_tensor = compare_models(model_funcs, model_args, testdata_loader)
print(error_tensor)

plot_error_statistics(("Bayesian Network", "MLP"), error_tensor, "Error Statistics for BN v.s. MLP")