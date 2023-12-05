import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.base import DAG
#G = DAG()

"""
Functions
"""


def ind_to_feature(ind):
    """
    returns the feature str associated with given index
    """
    features = ['windSpeed','cloudCover','windBearing','precipIntensity','dewPoint','pressure','apparentTemperature','visibility','humidity']
    return features[ind]

def feature_to_ind(feature):
    """
    returns the index associated with the given feature str
    """
    features = ['windSpeed','cloudCover','windBearing','precipIntensity','dewPoint','pressure','apparentTemperature','visibility','humidity']
    ind = features.index(feature)
    return ind

def create_network_from_edges(edge_tuples,
                              cpds,
                              network):
    
    #b_est = BayesianEstimator(template_model, train_df)
    #model_parameters = b_est.get_parameters(prior_type="BDeu",equivalent_sample_size=5,pseudo_counts=None,n_jobs=-1,weighted=False)

    nodes = []
    for ele in edge_tuples:
        if ele[0] not in nodes:
            nodes.append(ele[0])
        if ele[1] not in nodes:
            nodes.append(ele[1])
    print("nodes", nodes)
    for node in nodes:
        network.add_node(node)
    print(cpds)
    for link in edge_tuples:
        network.add_edge(link[0], link[1])

    network.add_cpds(*cpds)
    network.check_model()

    return network, nodes


"""

# Classes

class newDAG(DAG):
    def __init__(self, ebunch=None, latents=set()):
        super(DAG, self).__init__(ebunch)
        self.latents = set(latents)
        cycles = []
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            pass
        else:
            out_str = "Cycles are not allowed in a DAG."
            out_str += "\nEdges indicating the path taken for a loop: "
            out_str += "".join([f"({u},{v}) " for (u, v) in cycles])
            raise ValueError(out_str)
    
    def getDAG(self):
        dag = DAG()
        dag.add_edges_from(self.ebunch)
        return dag

"""

