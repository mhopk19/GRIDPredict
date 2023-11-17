import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import classes.powerdf as pdf
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_HomeC.csv", low_memory = False)

# fill missing values in
df.fillna(value = 0)
train_df = df.iloc[:100]
test_df = df.iloc[4000:4400]
df = df.iloc[:]

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

template_model = BayesianNetwork([('cloudCover','gen_Sol'),('windBearing','gen_Sol')])

# Create Bayesian Estimator object to learn the model parameters from the data
b_est = BayesianEstimator(template_model, train_df)
model_parameters = b_est.get_parameters(prior_type="BDeu",equivalent_sample_size=5,pseudo_counts=None,n_jobs=-1,weighted=False)

# all the conditional probability distributions from the data
cpd_cloudCover = model_parameters[0]
cpd_windBearing = model_parameters[2]
cpd_genSol = model_parameters[1]

learned_model = BayesianNetwork()
learned_model.add_node("cloudCover")
learned_model.add_node("windBearing")
learned_model.add_node("gen_Sol")

learned_model.add_edge("cloudCover","gen_Sol")
learned_model.add_edge("windBearing","gen_Sol")

learned_model.add_cpds(cpd_cloudCover, cpd_genSol, cpd_windBearing)
learned_model.check_model()

solver = VariableElimination(learned_model)

conditional_result = solver.query(variables=['gen_Sol'], evidence={'cloudCover':0.75, 'windBearing': 282.0})

# Testing

errors = np.array([])
gt_values = np.array([])
for i,val in enumerate(range(4000,4400)):
    #print(i, val)
    gt_solar = df.iloc[i]['gen_Sol']
    gt_cloudCover = df.iloc[i]['cloudCover']
    gt_windBearing = df.iloc[i]['windBearing']

    try:
        cpd = solver.query(variables=['gen_Sol'], evidence = {'cloudCover':gt_cloudCover,'windBearing':gt_windBearing})
    except KeyError:
        try:
            cpd = solver.query(variables=['gen_Sol'], evidence = {'cloudCover':gt_cloudCover})
        except KeyError:
            try:
                cpd = solver.query(variables=['gen_Sol'], evidence = {'windBearing':gt_windBearing})
            except:
                cpd = solver.query(variables=['gen_Sol'])

    queried_solar = float(cpd_sample(cpd, 'gen_Sol', samples=1)[0])
    gt_solar = float(df.iloc[i]['gen_Sol'])
    print("queried solar", queried_solar)
    print("gt solar", gt_solar)
    print("error", queried_solar - gt_solar)
    errors = np.append(errors, queried_solar - gt_solar)
    gt_values = np.append(gt_values, gt_solar)

RMSE = np.sqrt(np.mean(errors**2))

print("errors", errors)
print("RMSE error", RMSE)
print("RMSE percent of average value ", 100 * RMSE / np.mean(gt_values))


plt.plot(range(len(errors)), errors, 'b')
plt.ylabel('Errors (kW)')
plt.title("BN Solar Generation Estimate Error")
plt.show()

plt.plot([100,1000,2000,3000],[5.92,3.49,3.42,3.99],'b-o')
plt.xlabel("Training Set Time Instants")
plt.ylabel(r'RMSE $\%$ of avg.')
plt.title("Training Set Length v.s. Error")
plt.grid()
plt.show()