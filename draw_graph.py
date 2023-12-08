import networkx as nx
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1)
graph = nx.DiGraph([('visibility', 'pressure'), ('visibility', 'Microwave'), ('pressure', 'temperature'), ('pressure', 'Furnace'), ('pressure', 'Dishwasher'), ('pressure', 'Garage door'), ('pressure', 'Barn'), ('pressure', 'Fridge'), ('temperature', 'dewPoint'), ('temperature', 'gen_Sol'), ('temperature', 'Well'), ('dewPoint', 'windSpeed'), ('dewPoint', 'humidity'), ('windSpeed', 'Wine cellar'), ('precipIntensity', 'Living room')])
nx.draw(graph, with_labels=True,
        font_size = 6)
plt.show()