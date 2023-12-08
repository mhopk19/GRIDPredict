from itertools import combinations
import numpy as np
import networkx as nx
from sklearn.metrics import f1_score

import networkx as nx
import pylab as plt

from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

print("started...")

model = get_example_model("alarm")
samples = BayesianModelSampling(model).forward_sample(size=int(1e3))
samples.head()


def get_f1_score(estimated_model, true_model):
    nodes = estimated_model.nodes()
    est_adj = nx.to_numpy_matrix(
        estimated_model.to_undirected(), nodelist=nodes, weight=None
    )
    true_adj = nx.to_numpy_matrix(
        true_model.to_undirected(), nodelist=nodes, weight=None
    )

    f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))
    print("F1-score for the model skeleton: ", f1)

est = PC(data=samples)
estimated_model = est.estimate(variant="stable", max_cond_vars=4)
get_f1_score(estimated_model, model)
pc_graph = nx.DiGraph(estimated_model.edges())

scoring_method = K2Score(data=samples)
est = HillClimbSearch(data=samples)
estimated_model = est.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
get_f1_score(estimated_model, model)
print("estimated model", estimated_model)
print("class of estimated model", estimated_model.__class__)
hillclimb_graph = nx.DiGraph(estimated_model.edges())



fig, axs = plt.subplots(2)
fig.suptitle('Estimated Graph: (top) PC (bottom) Hillclimb-Search')

plt.axes(axs[0])
nx.draw(pc_graph, with_labels=True,
        font_size = 6)
plt.axes(axs[1])
nx.draw(hillclimb_graph, with_labels=True,
        font_size = 6)
    
plt.show()