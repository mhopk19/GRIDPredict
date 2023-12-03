# Import necessary libraries
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess data
def preprocess_data(data):
    return data.head(1000)

            
# def causal_discovery(data, maxlag, variables, alpha=0.05):
#     # Convert all columns to numeric and handle missing values
#     data = data[variables].apply(pd.to_numeric, errors='coerce')
#     data.dropna(inplace=True)  # You can also use data.fillna(method='ffill') or another method

#     # Check for Inf or NaN values
#     if data.isin([np.inf, -np.inf]).any().any():
#         raise ValueError("Data contains infinite values.")

#     potential_parents = {var: [] for var in variables}

#     # Fit a VAR model
#     model = VAR(data)
#     model_fitted = model.fit(maxlags=maxlag, ic='aic')

#     # Perform Granger causality tests
#     for var in variables:
#         for possible_cause in variables:
#             if var != possible_cause:
#                 test_result = grangercausalitytests(data[[var, possible_cause]], maxlag=maxlag, verbose=False)
#                 p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
#                 min_p_value = min(p_values)
#                 if min_p_value < alpha:
#                     potential_parents[var].append(possible_cause)

#     return potential_parents

def causal_discovery(data, maxlag, weather_variables, pred_variables, appliance_variables, alpha=0.05):
    # Convert all columns to numeric and handle missing values
    data[weather_variables] = data[weather_variables].apply(pd.to_numeric, errors='coerce')
    data[appliance_variables] = data[appliance_variables].apply(pd.to_numeric, errors='coerce')
    data[pred_variables] = data[pred_variables].apply(pd.to_numeric, errors='coerce')

    # Initialize dictionaries for potential parents
    potential_parents_for_pred = {var: [] for var in pred_variables}
    potential_parents_for_appliance = {var: [] for var in appliance_variables}

    # Perform Granger causality tests for prediction variables against weather and appliance variables
    for pred in pred_variables:
        for weather in weather_variables:
            test_result = grangercausalitytests(data[[pred, weather]], maxlag=maxlag, verbose=False)
            p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
            if min(p_values) < alpha:
                potential_parents_for_pred[pred].append(weather)
        for app in appliance_variables:
            test_result = grangercausalitytests(data[[pred, app]], maxlag=maxlag, verbose=False)
            p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
            if min(p_values) < alpha:
                potential_parents_for_pred[pred].append(app)

    # Perform Granger causality tests for appliance variables against weather variables
    for app in appliance_variables:
        for weather in weather_variables:
            test_result = grangercausalitytests(data[[app, weather]], maxlag=maxlag, verbose=False)
            p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
            if min(p_values) < alpha:
                potential_parents_for_appliance[app].append(weather)

    return potential_parents_for_pred, potential_parents_for_appliance


def define_node_order(potential_parents):
    # Initialize parent_counts for all nodes (including potential parents)
    all_nodes = set(potential_parents.keys())
    for parents in potential_parents.values():
        all_nodes.update(parents)
    parent_counts = {node: 0 for node in all_nodes}
    
    # Count how often each node appears as a potential parent
    for parents in potential_parents.values():
        for parent in parents:
            parent_counts[parent] += 1
    
    # Sort nodes based on the count, in decreasing order
    sorted_nodes = sorted(parent_counts, key=parent_counts.get, reverse=True)
    
    return sorted_nodes

# def define_node_order(potential_parents):
#     parent_counts = {node: 0 for node in potential_parents}
    
#     # Count how often each node appears as a potential parent
#     for node, parents in potential_parents.items():
#         for parent in parents:
#             parent_counts[parent] += 1
    
#     # Sort nodes based on the count, in decreasing order
#     sorted_nodes = sorted(parent_counts, key=parent_counts.get, reverse=True)
    
#     return sorted_nodes

# Modified function to learn structure using K2 algorithm
# def learn_structure_k2(data, node_order, max_parents, potential_parents):
    # k2 = K2Score(data)
    # parents = {node: set() for node in node_order}

    # for node in node_order:
    #     best_score = float("-inf")
    #     best_candidate = None
    #     candidates = set(potential_parents[node])

    #     while candidates and len(parents[node]) < max_parents:
    #         for candidate in candidates:
    #             test_parents = parents[node] | {candidate}
    #             test_edges = [(parent, node) for parent in test_parents]
    #             test_model = BayesianNetwork(test_edges)
    #             score = k2.score(test_model)

    #             if score > best_score:
    #                 best_score = score
    #                 best_candidate = candidate
            
    #         if best_candidate:
    #             parents[node].add(best_candidate)
    #             candidates.remove(best_candidate)
    #             best_candidate = None
    #         else:
    #             break

    # edges = [(parent, child) for child, parent_set in parents.items() for parent in parent_set]
    # model = BayesianNetwork(edges)
    # return model

def learn_structure_k2(data, node_order, max_parents, potential_parents):
    k2 = K2Score(data)
    parents = {node: set() for node in node_order}

    for node in node_order:
        best_score = float("-inf")
        best_candidate = None
        # Ensure node has an entry in potential_parents, even if it's an empty set
        candidates = set(potential_parents.get(node, []))

        while candidates and len(parents[node]) < max_parents:
            for candidate in candidates:
                test_parents = parents[node] | {candidate}
                test_edges = [(parent, node) for parent in test_parents]
                test_model = BayesianNetwork(test_edges)
                score = k2.score(test_model)

                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                parents[node].add(best_candidate)
                candidates.remove(best_candidate)
                best_candidate = None
            else:
                break

    edges = [(parent, child) for child, parent_set in parents.items() for parent in parent_set]
    model = BayesianNetwork(edges)
    return model


# Function to learn parameters of the model
def learn_parameters(model, data):
    model.fit(data, estimator=MaximumLikelihoodEstimator)

# Function to make predictions
def make_predictions(model, evidence):
    infer = VariableElimination(model)
    evidence_keys = list(evidence.keys())
    for key in evidence_keys:
        if key not in model.nodes():
            del evidence[key]
    query_vars = ['use_HO', 'gen_Sol']
    query_vars = [var for var in query_vars if var in model.nodes()]
    if not query_vars:
        return None
    return infer.query(variables=query_vars, evidence=evidence)

# # Main function to run the pipeline
# def main():
#     file_path = './cleaned_HomeC.csv'
#     data = load_data(file_path)
#     data = preprocess_data(data)
#     weather_variables = ['temperature', 'humidity', 'visibility', 'pressure', 'windSpeed', 'cloudCover', 'precipIntensity', 'dewPoint']
#     appliance_variables = [ 'Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 'Garage door', 'Barn', 'Well', 'Microwave', 'Living room', 'Furnace', 'Kitchen']
#     pred_variables = ['use_HO', 'gen_Sol']

#     # Apply causal discovery to identify potential parents
#     potential_parents = causal_discovery(data, maxlag = 5, variables = variables, alpha=0.05)
#     node_order = define_node_order(potential_parents)

#     # Modify this to provide node order and max_parents
#     model = learn_structure_k2(data, node_order, max_parents=10, potential_parents=potential_parents)
#     print("Learned Model Structure: ", model.edges())

#     learn_parameters(model, data)

#     evidence = {'time': '2016-01-01 05:00:00'}
#     predictions = make_predictions(model, evidence)
#     print(predictions)

# # Run the pipeline
# if __name__ == "__main__":
#     main()

def main():
    file_path = './cleaned_HomeC.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    weather_variables = ['temperature', 'humidity', 'visibility', 'pressure', 'windSpeed', 'cloudCover', 'precipIntensity', 'dewPoint']
    appliance_variables = ['Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 'Garage door', 'Barn', 'Well', 'Microwave', 'Living room', 'Furnace', 'Kitchen']
    pred_variables = ['use_HO', 'gen_Sol']
    # Select only the specified variables
    selected_variables = weather_variables + appliance_variables + pred_variables
    data = data[selected_variables]

    # Apply causal discovery to identify potential parents
    potential_parents_for_pred, potential_parents_for_appliance = causal_discovery(data, maxlag=5, weather_variables=weather_variables, pred_variables=pred_variables, appliance_variables=appliance_variables, alpha=0.05)

    # Merge potential parents dictionaries
    potential_parents = {**potential_parents_for_pred, **potential_parents_for_appliance}

    # Define the node order
    node_order = define_node_order(potential_parents)

    # Learn the structure using K2 algorithm
    model = learn_structure_k2(data, node_order, max_parents=10, potential_parents=potential_parents)
    print("Learned Model Structure: ", model.edges())

    learn_parameters(model, data)

    # Example evidence (modify as needed)
    evidence = {'temperature': 20}  # Example, replace with actual variable and value
    predictions = make_predictions(model, evidence)
    print(predictions)

# Run the pipeline
if __name__ == "__main__":
    main()