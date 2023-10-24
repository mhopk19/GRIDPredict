import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import classes.powerdf as pdf

df = pd.read_csv("cleaned_HomeC.csv", low_memory = False)

# fill missing values in
df.fillna(value = 0)
df = df.iloc[:400]
print(len(df))
print(df)

# USEFUL FUNCTIONS

def print_cpd(cpd,variable):
    for ind,state in enumerate(cpd.state_names[variable]):
        print("Probability of {} being {}".format(variable,state), cpd.values[ind])

def cpd_sample(cpd, variable, samples=1):
    values = np.array(cpd.state_names[variable])
    probs = np.array(cpd.values)
    return np.random.choice(values,samples, True, probs)


# EXAMPLE

template_model = BayesianModel([('cloudCover','gen_Sol'),('windBearing','gen_Sol')])

# Create Bayesian Estimator object to learn the model parameters from the data
b_est = BayesianEstimator(template_model,df)
model_parameters = b_est.get_parameters(prior_type="BDeu",equivalent_sample_size=5,pseudo_counts=None,n_jobs=-1,weighted=False)

# all the conditional probability distributions from the data
cpd_cloudCover = model_parameters[0]
cpd_windBearing = model_parameters[2]
cpd_genSol = model_parameters[1]

learned_model = BayesianModel()
learned_model.add_node("cloudCover")
learned_model.add_node("windBearing")
learned_model.add_node("gen_Sol")

learned_model.add_edge("cloudCover","gen_Sol")
learned_model.add_edge("windBearing","gen_Sol")

learned_model.add_cpds(cpd_cloudCover, cpd_genSol, cpd_windBearing)
learned_model.check_model()

solver = VariableElimination(learned_model)


result = solver.query(variables=['cloudCover'])
for ind,state in enumerate(result.state_names['cloudCover']):
    print("cloudCover probability of being {}".format(state), result.values[ind])

print_cpd(result, 'cloudCover')

# 10 samples from the cpd of gen_sol given that the cloudCover is 0.75

conditional_result = solver.query(variables=['gen_Sol'], evidence={'cloudCover':0.75, 'windBearing': 282.0})

print_cpd(conditional_result, 'gen_Sol')

# 10 samples from the cpd of gen_sol given that the cloudCover is 0.75

print("samples from the conditional probability distribution", cpd_sample(conditional_result, 'gen_Sol',10))