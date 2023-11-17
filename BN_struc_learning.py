# Import necessary libraries
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess data
def preprocess_data(data):
    # Selecting specific columns and limit to first 1000 rows for demo purposes
    # selected_columns = ['time', 'visibility', 'temperature', 'pressure', 'windSpeed', 'use_HO', 'gen_Sol']
    return data.head(10000)

# Function to learn structure of the Bayesian Network
def learn_structure(data):
    # Create a BicScore object
    bic = BicScore(data)
    # Initialize HillClimbSearch without scoring_method in constructor
    hc = HillClimbSearch(data)
    # Use the BicScore object with the estimate method
    best_model = BayesianNetwork(hc.estimate(scoring_method=bic).edges())
    return best_model

# Function to learn parameters of the model
def learn_parameters(model, data):
    model.fit(data, estimator=MaximumLikelihoodEstimator)

# Function to make predictions
def make_predictions(model, evidence):
    infer = VariableElimination(model)

    # Update evidence if necessary
    evidence_keys = list(evidence.keys())
    for key in evidence_keys:
        if key not in model.nodes():
            del evidence[key]

    # Check if the variables are in the model before querying
    query_vars = ['use_HO', 'gen_Sol']
    query_vars = [var for var in query_vars if var in model.nodes()]

    # If no variables to query, return None or handle as needed
    if not query_vars:
        return None  # or handle as needed

    return infer.query(variables=query_vars, evidence=evidence)

# Main function to run the pipeline
def main():
    file_path = './cleaned_HomeC.csv'  # Update with your file path
    data = load_data(file_path)
    data = preprocess_data(data)

    model = learn_structure(data)
    print("Learned Model Structure: ", model.edges())

    learn_parameters(model, data)

    # # Example evidence - update with actual values
    # evidence = {'time': '2016-01-01 05:00:00'}  
    # predictions = make_predictions(model, evidence)
    # print(predictions)

# Run the pipeline
if __name__ == "__main__":
    main()
