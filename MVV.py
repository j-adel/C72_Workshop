"""
To activate local environment, open C72_Workshop directory as folder
and execute the following command in Terminal: .\env\Scripts\activate

To run the file, execute: python MVV.py
"""

################### UTILITY #########################
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def format_name(name, limit):
    name = name.strip()    # remove empty space at start and end if exists
    if len(name) > limit:
        return name[:limit] + '..'
    else:
        return name

def append_csv_to_routes(csv_file, routes,route_names):
    df = pd.read_csv(csv_file)
    new_routes = df.values.tolist()
    
    for i in range(len(new_routes)):
        route_names.append(new_routes[i].pop(0))
        new_routes[i] = new_routes[i][0].split(',')
        new_routes[i] = [format_name(station,10) for station in new_routes[i]]
    
    routes.extend(new_routes)
    return routes, route_names

##################### GRAPH/NETWOK CONSTRUCTION ####################
routes = []
route_names = []
G = nx.Graph()

append_csv_to_routes('UBahn.csv',routes,route_names)
append_csv_to_routes('SBahn.csv',routes,route_names)
append_csv_to_routes('Trams.csv',routes,route_names)

for route in routes:
    G.add_nodes_from(route)

for route in routes:
    for i in range(len(route)-1):
        G.add_edge(route[i],route[i+1])

print('Number of nodes (stations):', len(G.nodes()))
print('Number of edges (Connections): ', len(G.edges))
############### DEGREE ANALYSIS ###########################

print('Graph Density: ', nx.density(G))
# # print nodes in G
print("\nNodes of G: ",G.nodes())
# degree_distribution = sorted((n,d) for n, d in G.degree())

# print("\ndegree_distribution:\n",degree_distribution)

clustering = nx.clustering(G)
filtered_clustering = {node: coeff for node, coeff in clustering.items() if coeff > 0}
print("\nClustering:\n", filtered_clustering)

diameter = nx.diameter(G)
print("\nDiameter:",diameter)

paths_lengths = dict(nx.all_pairs_shortest_path_length(G))
# make histogram for number of paths of given lengths
lengths_distribution = {}
for origin in paths_lengths:
    for destination in paths_lengths[origin]:
        length = paths_lengths[origin][destination]
        if length not in lengths_distribution:
            lengths_distribution[length] = 1
        else:
            lengths_distribution[length] += 1

print("\nDistance_distribution:\n", lengths_distribution)

############### CENTRALITY ANALYSIS ###########################

# Compute closeness centrality
closeness_centrality = nx.closeness_centrality(G)
sorted_closeness_centrality = sorted(closeness_centrality.items(), key=lambda x:x[1], reverse=True)
print("\nCloseness Centrality:", sorted_closeness_centrality[0:20])

# Compute betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)
sorted_betweenness_centrality = sorted(betweenness_centrality.items(), key=lambda x:x[1], reverse=True)
print("Betweenness Centrality:", sorted_betweenness_centrality[0:20])

# Compute link betweenness centrality
edge_betweenness_centrality = nx.edge_betweenness_centrality(G)
sorted_edge_betweenness_centrality = sorted(edge_betweenness_centrality.items(), key=lambda x:x[1], reverse=True)
print("Edge Betweenness Centrality:", sorted_edge_betweenness_centrality[0:20])

import matplotlib.colors as mcolors
# Normalize edge betweenness values for color mapping
max_centrality = max(edge_betweenness_centrality.values())
min_centrality = min(edge_betweenness_centrality.values())
normalized_centrality = {e: (c - min_centrality) / (max_centrality - min_centrality) for e, c in edge_betweenness_centrality.items()}

# Create a color map
cmap = plt.get_cmap('viridis')

# Draw the network
positions = nx.kamada_kawai_layout(G)

# Draw edges with colors based on edge betweenness centrality
nx.draw_networkx_edges(G, positions, edge_color=[cmap(normalized_centrality[e]) for e in G.edges()], width=2)

# Show plot
plt.show()

# Normalize closeness centrality values for color mapping
max_closeness = max(closeness_centrality.values())
min_closeness = min(closeness_centrality.values())
normalized_closeness = {node: (centrality - min_closeness) / (max_closeness - min_closeness) for node, centrality in closeness_centrality.items()}

########### VISUALISATION #############
import matplotlib.colors as mcolors
# Normalize edge betweenness values for color mapping
max_centrality = max(edge_betweenness_centrality.values())
min_centrality = min(edge_betweenness_centrality.values())
normalized_centrality = {e: (c - min_centrality) / (max_centrality - min_centrality) for e, c in edge_betweenness_centrality.items()}

# Create a color map
cmap = plt.get_cmap('viridis')

# Draw the network
positions = nx.kamada_kawai_layout(G)

# Draw edges with colors based on edge betweenness centrality
nx.draw_networkx_edges(G, positions, edge_color=[cmap(normalized_centrality[e]) for e in G.edges()], width=2)

# Show plot
plt.show()




# Normalize closeness centrality values for color mapping
max_closeness = max(closeness_centrality.values())
min_closeness = min(closeness_centrality.values())
normalized_closeness = {node: (centrality - min_closeness) / (max_closeness - min_closeness) for node, centrality in closeness_centrality.items()}

# Create a color map
cmap = plt.get_cmap('plasma')  # Using a different colormap for distinction

# Draw the network
# Draw nodes with colors based on closeness centrality
nx.draw_networkx_nodes(G, positions, node_size=80, cmap=cmap, node_color=[normalized_closeness[node] for node in G.nodes()])
# nx.draw_networkx_labels(G, positions, font_size=6)
nx.draw_networkx_edges(G, positions, edge_color='gray', width=1)  # Drawing edges in gray for better contrast

plt.show()