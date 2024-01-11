import networkx as nx
import matplotlib.pyplot as plt
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


def get_component_containing(G, node):
    """
    returns: isolated subgraph containing "node"
    """
    UG = G.to_undirected()
    sub_graphs = nx.connected_components(UG)
    for graph in sub_graphs:
        print("graph", graph)
        for vert in graph:
            print("vertex", vert)
            if (str(vert) == str(node)):
                subgraph = G.subgraph(graph)
                return list (subgraph.edges)



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


def draw_graph(nx_graph):
    fig, axs = plt.subplots(1)
    nx.draw(graph, with_labels=True,
        font_size = 6)
    plt.show()


if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_nodes_from([1,2,3,4])
    G.add_edge(1,2)
    G.add_edge(3,4)
    G.add_edge(2,5)

    # make an undirected copy of the digraph
    UG = G.to_undirected()

    x = get_component_containing(G, "1")
    # extract subgraphs
    print(x)