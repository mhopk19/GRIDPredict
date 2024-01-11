import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import sys, os
sys.path.append("./..")
import classes.powerdf as pdf
import matplotlib.pyplot as plt
import classes.powerdf as pdf
import classes.utils as utils
import matplotlib.pyplot as plt
from classes.data_loader import smarthome_dataset
import copy
import networkx as nx
import time
import sys

cur_path = os.path.abspath(os.getcwd())
dataset_path = os.path.join(cur_path,"cleaned_HomeC.csv")
print(dataset_path)
df = pd.read_csv(dataset_path, low_memory = False)
dataset = smarthome_dataset("cleaned_HomeC.csv" , dayahead = False)
print("dataset length", len(dataset))

# fill missing values in
df.fillna(value = 0)
train_df = df.iloc[:750]
test_df = df.iloc[50000:51000]
df = df.iloc[:]
print(len(df))


"""
Testing Different Bayesian Network Structures
"""

def print_cpd(cpd,variable):
    for ind,state in enumerate(cpd.state_names[variable]):
        print("Probability of {} being {}".format(variable,state), cpd.values[ind])

def cpd_sample(cpd, variable, samples=1):
    values = np.array(cpd.state_names[variable])
    probs = np.array(cpd.values)
    print("values", values)
    print("probs", probs)
    return np.random.choice(values,samples, True, p = probs)


# EXAMPLE

og_template_model = BayesianNetwork([('visibility', 'pressure'), ('visibility', 'Microwave'), ('pressure', 'temperature'), ('pressure', 'Fridge'), ('pressure', 'Furnace'), ('pressure', 'Garage door'), ('pressure', 'Dishwasher'), ('pressure', 'Barn'), ('temperature', 'Well'), ('temperature', 'gen_Sol'), ('windSpeed', 'Wine cellar'), ('humidity', 'use_HO'), ('dewPoint', 'humidity'), ('dewPoint', 'windSpeed'), ('precipIntensity', 'Living room')])
# 1000 data points
og_network_edges = [('visibility', 'pressure'), ('visibility', 'Microwave'), ('pressure', 'temperature'), ('pressure', 'Fridge'), ('pressure', 'Furnace'), ('pressure', 'Garage door'), ('pressure', 'Dishwasher'), ('pressure', 'Barn'), ('temperature', 'Well'), ('temperature', 'gen_Sol'), ('windSpeed', 'Wine cellar'), ('humidity', 'use_HO'), ('dewPoint', 'humidity'), ('dewPoint', 'windSpeed'), ('precipIntensity', 'Living room')]
# 10,000 (focused weather variables)
og_network_edges = [('windSpeed', 'pressure'), ('pressure', 'dewPoint'), ('pressure', 'temperature'), ('pressure', 'gen_Sol'), ('dewPoint', 'visibility'), ('visibility', 'precipIntensity')]
# 100,000 (focused weather variables)
og_network_edges = [('windSpeed', 'pressure'), ('pressure', 'temperature'), ('pressure', 'gen_Sol'), ('cloudCover', 'dewPoint'), ('cloudCover', 'precipIntensity'), ('cloudCover', 'visibility'), ('humidity', 'cloudCover')]

# pc algorithm 10000
og_network_edges = [('dewPoint', 'precipIntensity'), ('cloudCover','gen_Sol'),('dewPoint', 'cloudCover'), ('cloudCover', 'precipIntensity'), ('cloudCover', 'temperature'), ('humidity', 'precipIntensity'), ('humidity', 'cloudCover'), ('visibility', 'precipIntensity'), ('visibility', 'cloudCover'), ('temperature', 'precipIntensity'), ('pressure', 'cloudCover'), ('pressure', 'precipIntensity'), ('pressure', 'dewPoint'), ('windSpeed', 'cloudCover'), ('windSpeed', 'precipIntensity'), ('windSpeed', 'dewPoint'), ('windSpeed', 'pressure')] 
# gs algorithm 10000
#og_network_edges = [('temperature', 'humidity'), ('temperature', 'visibility'), ('temperature', 'pressure'), ('temperature', 'windSpeed'), ('temperature', 'cloudCover'), ('temperature', 'precipIntensity'), ('temperature', 'dewPoint'), ('temperature', 'gen_Sol'), ('humidity', 'visibility'), ('humidity', 'pressure'), ('humidity', 'windSpeed'), ('humidity', 'cloudCover'), ('humidity', 'precipIntensity'), ('humidity', 'dewPoint'), ('humidity', 'gen_Sol'), ('visibility', 'pressure'), ('visibility', 'windSpeed'), ('visibility', 'cloudCover'), ('visibility', 'dewPoint'), ('pressure', 'windSpeed'), ('pressure', 'cloudCover'), ('pressure', 'precipIntensity'), ('pressure', 'dewPoint'), ('pressure', 'gen_Sol'), ('windSpeed', 'cloudCover'), ('windSpeed', 'precipIntensity'), ('windSpeed', 'dewPoint'), ('windSpeed', 'gen_Sol'), ('cloudCover', 'precipIntensity'), ('cloudCover', 'dewPoint'), ('cloudCover', 'gen_Sol'), ('precipIntensity', 'dewPoint'), ('dewPoint', 'gen_Sol')]

#k2
#og_network_edges = [('humidity', 'temperature'), ('humidity', 'dewPoint'), ('humidity', 'pressure'), ('humidity', 'windSpeed'), ('dewPoint', 'temperature'), ('pressure', 'temperature'), ('pressure', 'dewPoint'), ('windSpeed', 'temperature'), ('windSpeed', 'dewPoint'), ('windSpeed', 'pressure'), ('visibility', 'temperature'), ('visibility', 'dewPoint'), ('visibility', 'pressure'), ('visibility', 'windSpeed'), ('visibility', 'humidity'), ('visibility', 'cloudCover'), ('cloudCover', 'temperature'), ('cloudCover', 'dewPoint'), ('cloudCover', 'pressure'), ('cloudCover', 'windSpeed'), ('cloudCover', 'humidity'), ('precipIntensity', 'temperature'), ('precipIntensity', 'dewPoint'), ('precipIntensity', 'pressure'), ('precipIntensity', 'windSpeed'), ('precipIntensity', 'humidity'), ('precipIntensity', 'cloudCover'), ('precipIntensity', 'visibility'), ('gen_Sol', 'temperature'), ('gen_Sol', 'dewPoint'), ('gen_Sol', 'pressure'), ('gen_Sol', 'windSpeed'), ('gen_Sol', 'humidity'), ('gen_Sol', 'cloudCover'), ('gen_Sol', 'visibility')]

start_time = time.time()
network_edges = utils.get_component_containing(nx.DiGraph(og_network_edges), "gen_Sol")
G = nx.DiGraph(network_edges)
print("old graph size", G.size())
import copy
new_G = copy.deepcopy(G)
for x in G.nodes():
    if (x=="gen_Sol"):
        continue
    if (nx.all_simple_paths(new_G, source=x, target='gen_Sol', cutoff = 10) == []):
        new_G.remove_node(x)
        continue
    if (new_G.out_degree(x)==0 and new_G.in_degree(x)==1):
        new_G.remove_node(x)
#[G.remove_node(x) for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
network_edges = [e for e in new_G.edges]
print("new graph size", new_G.size())
print("network edges", network_edges)
print("weather isolated edges", network_edges)
template_model = BayesianNetwork(network_edges)

#display(df)
#print(df["visibility"])

# Create Bayesian Estimator object to learn the model parameters from the data
b_est = BayesianEstimator(template_model, train_df)
m_est = MaximumLikelihoodEstimator(template_model, train_df)
#model_parameters = b_est.get_parameters(prior_type="BDeu",equivalent_sample_size=1,pseudo_counts=1,n_jobs=1,weighted=False)
model_parameters = m_est.get_parameters(n_jobs=1, weighted=False)

cpds = model_parameters
learned_model, nodes = utils.create_network_from_edges(network_edges, cpds, BayesianNetwork())

solver = VariableElimination(learned_model)

end_time = time.time()
print("time training", end_time - start_time)
#time.sleep(5000)

#sys.exit(1)

# Testing

trials = 2
RMSE_vector = trials * [0]

timed = False

for t in range(trials): 
    errors = np.array([])
    gt_values = np.array([])
    for i,val in enumerate(range(50000+t*1000,51000+t*1000)):
        variable_values = []
        evidence_nodes = copy.copy(nodes)
        evidence_nodes.remove('gen_Sol')
        evidence_nodes = nodes
        for node in evidence_nodes:
            variable_values.append((node, df.iloc[i][node]))
    
        if (timed == False):
            start_time = time.time()
        queried = False
        while (queried == False):
            evidence_dict = {}
            for ele in variable_values:
                evidence_dict[ele[0]] = ele[1]
            try:
                evidence_dict = {}
                for ele in variable_values:
                    evidence_dict[ele[0]] = ele[1]
                cpd = solver.query(['gen_Sol'], evidence = evidence_dict)
                queried = True
            except:
                variable_values.pop()
                if (variable_values == []):
                    cpd = solver.query(variables=['gen_Sol'])
                    queried = True
            
            if (timed == False):
                end_time = time.time()
                print("exec time", end_time - start_time)
                time.sleep(5)
                timed = True

        queried_solar = float(cpd_sample(cpd, 'gen_Sol', samples=1)[0])
        gt_solar = float(df.iloc[i+10]['gen_Sol'])
        print("queried solar", queried_solar)
        #print("gt solar", gt_solar)
        #print("error", queried_solar - gt_solar)
        errors = np.append(errors, queried_solar - gt_solar)
        gt_values = np.append(gt_values, gt_solar)
        print("iteration", i)

    RMSE = (np.sqrt(np.mean(np.square((errors) / gt_values)))) * 100
    MAE = (np.sum(np.abs(errors) / gt_values) * 100) / len(gt_values)

    RMSE_vector[t] = np.mean(errors)
    #print("errors", errors)
    #print("mean error", np.mean(errors))
    #print("RMSE error", RMSE)
    #print("RMSE percent of average value ", 100 * RMSE / np.mean(gt_values))
    #print("MAE percent error", MAE)
    
total_mean = np.mean(RMSE_vector)
print("total mean {}".format(total_mean))