from graphviz import Source
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

#path = './Full_DFA_Sokoban_6_6.dot'
#path = './Full-DFA-Hanoi_4_4.dot'

path = './Full-DFA_blocks4Colors.dot'

matplotlib.use("Agg") 

G = nx.nx_agraph.read_dot(path)




def path_to_edges(path):
    return [[path[i], path[i+1]] for i in range(len(path) - 1)]



def save_stuff(dire, array, file_name):
    data = {
        "stuff": array,
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = str(file_name)+".p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)




def return_transitions_to_remove(
    dir_saving_name = "blocks4_missing_edges_2",
    alternative_of_at_least_N_states = 8
    ):


    # 
    edge_to_test_on = None
    alternative_path = []
    nb_nodes_of_max_path = 0

    #alternative_of_at_least_N_states = 11 # hanoi

    #alternative_of_at_least_N_states = 13 # blocks


    ### loop over all the edges of the graph until it finds an edge for which
    # a plan of at least alternative_of_at_least_N_states states (10 steps) exists !
    # when found, save the <=> plan and edge in edge_to_test_on and alternative_path
    for edge in G.edges():

        print(list(edge)[1])
        print(type(list(edge)[1]))


        #if 'fake' not in edge and ( list(edge)[0] != '300000000001000000000000000000000000' and list(edge)[1] != '300001000000000000000000000000000000' ) and ( list(edge)[0] != '402000000000000000000000000000001000' and list(edge)[1] != '402000000000000000000000000000010000' ):
        if 'fake' not in edge and ( list(edge)[0] == '([[], [b], [c], [d]], a)' and list(edge)[1] == '([[], [b], [c], [d, a]], )' ):

            paths = nx.all_simple_paths(G, edge[0], edge[1], cutoff=alternative_of_at_least_N_states-1)

            max_length = 0 # length in #nodes
            max_path = None


            for p in paths:
                if len(p) > max_length:
                    max_length = len(p)
                    max_path = p

            # if, for current edge, we found an alternative plan of lengths > 9 (#nodes > 10)
            # we print it out and store the plan, the edges (init/goal), and the #nodes
            if max_length >= alternative_of_at_least_N_states:
                #print("edge {} has alternative path of length {}".format(str(edge), str(max_length)))
                edge_to_test_on = edge
                alternative_path = max_path
                nb_nodes_of_max_path = max_length
                # print("was here")
                # exit()
                #print("this path is {}".format(str(max_path)))
                break


    edges_of_alt_plan = path_to_edges(alternative_path)

    ### we retrieve all the plans that exist for the chosen edge 
    # up to the length of the alternative plan (i.e. max_length (#nodes) - 1)
    all_paths_btween_chosen_edges = nx.all_simple_paths(G, edge_to_test_on[0], edge_to_test_on[1], cutoff=max_length-1)

    #all_paths_btween_chosen_edges_list = [pa for pa in all_paths_btween_chosen_edges]

    G_short = G.copy()

    removed_edges = []

    #print("nber of all_paths_btween_chosen_edges {}".format(str(len(all_paths_btween_chosen_edges_list))))

    ### we go over all these plans and for each
    # we remove an edge that is not in the alternative plan
    for p in all_paths_btween_chosen_edges:
        
        for ed in path_to_edges(p):

            if ed not in edges_of_alt_plan and ed not in removed_edges:
                G_short.remove_edge(ed[0], ed[1])
                removed_edges.append(ed)
                break




    ###### Now, the testing
    ### on a deux nodes voisins 
    ### on a un alternative plan
    ### on a bloqués tous les autres plans alternatifs
    ### DONC on veut vérifier qu'effectivement le seul plan EST "alternative plan"


    print("edge to be tested is {}".format(str(edge_to_test_on)))

    print("alternative plan is {}".format(str(alternative_path)))

    print("#edges G before: {}".format(str(len(G.edges()))))

    print("#edges G after: {}".format(str(len(G_short.edges()))))


    # print(nx.shortest_path(G_short, edge_to_test_on[0], edge_to_test_on[1]))

    if not os.path.exists(dir_saving_name):
        os.makedirs(dir_saving_name) 
    exp_folder = dir_saving_name

    save_stuff(exp_folder, edge_to_test_on[0], "init_state")
    save_stuff(exp_folder, edge_to_test_on[1], "goal_state")
    save_stuff(exp_folder, removed_edges, "deleted_edges")



    return removed_edges



return_transitions_to_remove()