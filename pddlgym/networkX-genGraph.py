from graphviz import Source
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#path = './DFA-Hanoi_4_4.dot'

path = './Full_DFA_Sokoban_6_6.dot'
#path = './Full-DFA-Hanoi_4_4.dot'
#path = './Full-DFA_blocks4Colors.dot'


matplotlib.use("Agg") 

G = nx.nx_agraph.read_dot(path)

nodes = G.nodes()



# ################ GENERERATE GRAPH WITH A COLORED PATH ##################################

# #Compute the layout with increased spacing
# # 'k' is a parameter that affects the distance between nodes. Adjusting it might help.
# pos = nx.spring_layout(G, k=5.5*(1/np.sqrt(len(G.nodes()))), iterations=50)

# #path = ['Ed4E[d3d2d1]', 'Ed4d1[d3d2]', 'E[d4d2]d1d3', 'd3[d4d2]d1E', '[d3d2]d4d1E', '[d3d2]Ed1d4', 'd3d2d1d4', 'Ed2d1[d4d3]', 'EEd1[d4d3d2]', 'EEE[d4d3d2d1]']

# #path = ['420000100', '420100000', '420010000', '420001000', '421000000', '310000000', '300010000', '300000010', '300000100']

# path = ['([[], [], [], [d, c, b, a]], )', '([[], [], [], [d, c, b]], a)', '([[a], [], [], [d, c, b]], )', '([[a], [], [], [d, c]], b)', '([[a], [b], [], [d, c]], )', '([[a], [b], [], [d]], c)', '([[a], [b], [c], [d]], )', '([[a], [], [c], [d]], b)', '([[a], [], [c, b], [d]], )', '([[], [], [c, b], [d]], a)', '([[], [], [c, b, a], [d]], )', '([[], [], [c, b, a], []], d)', '([[], [], [c, b, a, d], []], )']



# # Create a list of edges in the path
# path_edges = list(zip(path[:-1], path[1:]))

# # Create edge colors and widths based on whether they are in the path
# edge_colors = []
# edge_widths = []
# for u, v in G.edges():
#     if (u, v) in path_edges or (v, u) in path_edges:
#         edge_colors.append('red')  # Color for path edges
#         edge_widths.append(2.5)    # Thicker edge width for path edges
#     else:
#         edge_colors.append('black') # Color for other edges
#         edge_widths.append(0.5)     # Normal edge width for other edges


# # Optionally scale the positions
# for key, value in pos.items():
#     pos[key] = value * 2  # Scaling factor to increase spacing

# # Create a figure
# fig = plt.figure(figsize=(50, 50))  # You can adjust the size of the figure to fit the nodes

# # Draw the graph components
# nx.draw_networkx_nodes(G, pos, node_color="#ff5733", alpha=0.)
# nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
# nx.draw_networkx_labels(G, pos, font_size=13)

# # Save the graph to a file
# plt.savefig("blocks4_longest_plan.png")



# ################# DISPLAY ALL PROBLEMS BY LENGTH (last are the longest) ########################################

# dico = {}

# for size in range(99):

#     for n in nodes:

#         if n != 'fake':
            
#             all_reachable_nodes = nx.descendants_at_distance(G, n, size)

#             if len(all_reachable_nodes) > 0:

#                 dico[size] = []

#                 for r in all_reachable_nodes:

#                     if r != 'fake':

#                         if [n, r] not in dico[size] and [n, r] not in dico[size]:
                            
#                             dico[size].append([n, r])

# print(dico)

# exit()


# s = Source.from_file(path)
# A.draw("k5.png", prog="neato")
# G = nx.DiGraph(s)

# nx.draw_networkx(G, with_labels=True, node_size=200, font_size=6)
# nx.spring_layout(G, k=50, iterations=100)

# fig = matplotlib.pyplot.figure()
# nx.draw(G, ax=fig.add_subplot())
# Save plot to file
#matplotlib.use("Agg") 
# plt.savefig("graph.png")








###############################
# DISPLAY optimal plan

shortest = nx.shortest_path(G, source='420000000000000000000000000000000001', target='401000000000000000000000000000000002', method='dijkstra')
print("shortest")
print(shortest)
exit()

###############################
# Display alternative plan

for i in range(len(shortest)-1):

    transi = [shortest[i], shortest[i+1]]
    

    for path in nx.all_simple_paths(G, source='([[], [], [], [d, c, b, a]], )', target='([[], [], [c, b, a, d], []], )', cutoff=13):
        if len(path) > 9:
            transi_in_path = False
            for j in range(len(path)-1):
                transi2 = [path[i], path[i+1]]
                if transi == transi2:
                    transi_in_path = True
            
            if not transi_in_path:
                print("found an alternative path without transition {}".format(str(transi)))
                print("alternative path is:")
                print(path)
                exit()
