from graphviz import Source
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pickle

#path = './DFA-Hanoi_4_4.dot'


parser = argparse.ArgumentParser(description="A script to generate traces from a domain")
parser.add_argument('--exp_folder', type=str, help='exp_folder name')
parser.add_argument('--domain', type=str, required=True)

args = parser.parse_args()





# create the "pbs"
problems_folder_name = args.exp_folder+"/pbs"
if not os.path.exists(problems_folder_name):
    os.makedirs(problems_folder_name) 


# Les sous folders, comment les créer ?

# 



path = ""
if args.domain == "hanoi":
    path = '/workspace/R-latplan/r_latplan_datasets/hanoi/Full-DFA-Hanoi_4_4.dot'
elif args.domain == "sokoban":
    path = '/workspace/R-latplan/r_latplan_datasets/sokoban/Full_DFA_Sokoban_6_6.dot'
elif args.domain == "blocks":
    path = '/workspace/R-latplan/r_latplan_datasets/blocks/Full-DFA_blocks4Colors.dot'



matplotlib.use("Agg") 

G = nx.nx_agraph.read_dot(path)

nodes = G.nodes()




def save_state(dire, state, init_or_goal = "init"):
    data = {
        "stuff": state,
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    if init_or_goal == "init":
        filename = "init_sym.p"
    else:
        filename = "goal_sym.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)



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



################# DISPLAY ALL PROBLEMS BY LENGTH (last are the longest) ########################################

def generate_symbolic_problems():


    three_longest_problems = {}

    dico = {}

    for size in range(99):

        for n in nodes:

            if n != 'fake':
                
                all_reachable_nodes = nx.descendants_at_distance(G, n, size)

                if len(all_reachable_nodes) > 0:

                    # the size is the distance from init to goal
                    dico[size] = []

                    for r in all_reachable_nodes:

                        if r != 'fake':

                            if [n, r] not in dico[size] and [n, r] not in dico[size]:
                                
                                dico[size].append([n, r])



    longests_1 = list(dico.items())[-1:]
    longest_prob = longests_1[0][0]

    
    # longest hanoi: 9 + 1 = 10 nodes  (distance mediane = 5)
    # longest sokoban: 20 + 1 = 21 nodes    (distance mediane = 10)
    # longest blocks: 12 + 1 = 13 nodes     (distance mediane = 6)


    liste = list(dico.items())



    index = 1

    cc = 0
    
    counter_pbs = 0
    done_ = False
    #### add in three_longest_problems the 3 longest ones

    while True:

        if done_:
            break

        if index == 1:
            probs = liste[-1:] # take the last pb of the whole list (the longest one)
        else:
            probs = liste[-index:-(index-1)] # take the "before" problems

        tmp_probs = probs[0]

        distance = tmp_probs[0]
        tmp_liste = tmp_probs[1]
        
        cc += 1
        
        while( len( tmp_liste ) > 0 ):

            if distance not in three_longest_problems:
                three_longest_problems[distance] = []

            if counter_pbs > 2:
                done_ = True
                #return three_longest_problems
                break

            three_longest_problems[distance].append(tmp_liste[0])
            counter_pbs += 1
    
            tmp_liste.pop(0)

        index += 1


    # longest hanoi: 9
    # longest sokoban: 20
    # longest blocks: 12

    # PBs médians !!!
    #print(dico[5]) # hanoi
    # print(dico[10]) # sokoban
    # print(dico[6]) # blocks

    # PBs du quart
    #print(dico[3]) # hanoi      OK
    #print(dico[5]) # sokoban    OK
    #print(dico[3]) # blocks      OK

    # One step problems
    #print(dico[1]) # hanoi      OK
    #print(dico[1]) # sokoban    OK
    #print(dico[1]) # blocks      OK


    ### put in three_median_problems the 3 median problems
    three_median_problems = {}
    dist = longest_prob // 2
    if args.domain == "hanoi":
        dist += 1
    three_median_problems[dist] = []
    for ii, pb in enumerate(dico[dist]):
        if ii > 2: break
        three_median_problems[dist].append(pb)    

    ### put in three_quarter_problems the 3 quarter problems
    three_quarter_problems = {}
    dist = longest_prob // 4
    if args.domain == "hanoi":
        dist += 1
    three_quarter_problems[dist] = []
    for ii, pb in enumerate(dico[dist]):
        if ii > 2: break
        three_quarter_problems[dist].append(pb) 

    ### put in three_1_step_problems the 3 1_step problems
    three_1_step_problems =  {}
    dist = 1
    three_1_step_problems[dist] = []
    for ii, pb in enumerate(dico[dist]):
        if ii > 2: break
        three_1_step_problems[dist].append(pb) 


    return [three_longest_problems, three_median_problems, three_quarter_problems, three_1_step_problems]



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

# shortest = nx.shortest_path(G, source='420000000000000000000000000000000001', target='401000000000000000000000000000000002', method='dijkstra')
# print("shortest")
# print(shortest)
# exit()

###############################
# Display alternative plan

# for i in range(len(shortest)-1):

#     transi = [shortest[i], shortest[i+1]]
    

#     for path in nx.all_simple_paths(G, source='([[], [], [], [d, c, b, a]], )', target='([[], [], [c, b, a, d], []], )', cutoff=13):
#         if len(path) > 9:
#             transi_in_path = False
#             for j in range(len(path)-1):
#                 transi2 = [path[i], path[i+1]]
#                 if transi == transi2:
#                     transi_in_path = True
            
#             if not transi_in_path:
#                 print("found an alternative path without transition {}".format(str(transi)))
#                 print("alternative path is:")
#                 print(path)
#                 exit()




# on va faire un sous rep par truc
# les 3 plus grand problems
# OU : un proiblem et LES NODES (mais ça c'est autre fichier)

if args.domain == "hanoi" or args.domain == "blocks" or args.domain == "sokoban":


    pbs_dicos = generate_symbolic_problems()

    #pbs_dicos = [pbs_dicos[-1]]

    for pbs_dico in pbs_dicos:


        # print("pbs_dico")
        # print(pbs_dico)
        for k, v in pbs_dico.items():
            ccc = 0
            if len(v) != 0:
                for pair in v:
                    problem_folder_name = args.exp_folder+"/pbs/"+str(k)+"_"+str(ccc)
                    if not os.path.exists(problem_folder_name):
                        os.makedirs(problem_folder_name) 
                    #save_pbs(problem_folder_name, pair[0], pair[1])
                    print("START for {}".format(problem_folder_name))
                    print("init {}".format(pair[0]))
                    print("goal {}".format(pair[1]))
                    print("FINISH")
                    save_state(problem_folder_name, pair[0], "init")
                    save_state(problem_folder_name, pair[1], "goal")
                    ccc += 1

    print("WAS HERE ")

    # #save_dataset(dataset_folder_name, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min)

