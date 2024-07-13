from graphviz import Source
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse



print("os.getcwd()")
print(os.getcwd())

parser = argparse.ArgumentParser(description="A script to generate the partial dfa")
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--exp_folder', type=str, help='exp_folder name')
#parser.add_argument('--exp_nb', type=str, help='exp_nb name')
# 
args = parser.parse_args()


# create the "pbs"
problems_folder_name = args.exp_folder+"/pbs"
if not os.path.exists(problems_folder_name):
    os.makedirs(problems_folder_name) 


#path = './Full_DFA_Sokoban_6_6.dot'
#path = './Full-DFA-Hanoi_4_4.dot'

path = ""



if args.domain == "blocks":
    path = os.getcwd()+"/r_latplan_datasets/"+str(args.domain)+"/"+'Full-DFA_blocks4Colors.dot'
elif args.domain == "sokoban":
    path = os.getcwd()+"/r_latplan_datasets/"+str(args.domain)+"/"+'Full_DFA_Sokoban_6_6.dot'
else:
    path = os.getcwd()+"/r_latplan_datasets/"+str(args.domain)+"/"+'Full-DFA-Hanoi_4_4.dot'



exp_folder = args.exp_folder


matplotlib.use("Agg") 

G = nx.nx_agraph.read_dot(path)




def path_to_edges(path):
    return [[path[i], path[i+1]] for i in range(len(path) - 1)]



def save_stuff(dire, array, file_name):
    data = {
        "stuff": array,
    }
    filename = str(file_name)+".p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)




def return_transitions_to_remove(
    alternative_of_at_least_N_states = 8,
    nodes_to_avoid = []
    ):


    # 
    edge_to_test_on = None
    alternative_path = []
    nb_nodes_of_max_path = 0


    ### loop over all the edges of the graph until it finds an edge for which
    # a plan of at least alternative_of_at_least_N_states states (10 steps) exists !
    # when found, save the <=> plan and edge in edge_to_test_on and alternative_path

    

    for edge in G.edges():

        if edge[0] in  nodes_to_avoid:
            continue

        print(list(edge)[1])
        print(type(list(edge)[1]))


        #if 'fake' not in edge and ( list(edge)[0] == '([[], [b], [c], [d]], a)' and list(edge)[1] == '([[], [b], [c], [d, a]], )' ):
        if 'fake' not in edge:

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
    len_alternative = len(alternative_path)

    return removed_edges, edge_to_test_on[0], edge_to_test_on[1], len_alternative





def get_and_split_subdirs(base_dir):

    subdirs = []
    
    # Iterate through the contents of the base directory
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        
        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            # Split the directory name by "_"
            split_name = entry.split("_")
            if len(split_name) > 1:
                subdirs.append(split_name[0])
    
    return subdirs



to_avoid = get_and_split_subdirs(args.exp_folder+"/pbs")


if args.domain == "hanoi":

    # nber steps _ first state _ 0
    if len(to_avoid) > 2:
        print("there are already three problems")
    
    else:

        removed_edges, edge_to_test_on_0, edge_to_test_on_1, len_  = return_transitions_to_remove(10, to_avoid)
        problem_folder_name = args.exp_folder+"/pbs/"+edge_to_test_on_0+"_"+str(len_)

        if not os.path.exists(problem_folder_name):
            os.makedirs(problem_folder_name) 

            save_stuff(problem_folder_name, edge_to_test_on_0, "init_sym")
            save_stuff(problem_folder_name, edge_to_test_on_1, "goal_sym")
            save_stuff(problem_folder_name, removed_edges, "deleted_edges_sym")

elif args.domain == "blocks":

    if len(to_avoid) > 2:
        print("there are already three problems")
    
    else:


        removed_edges, edge_to_test_on_0, edge_to_test_on_1, len_  = return_transitions_to_remove(12, to_avoid)
        problem_folder_name = args.exp_folder+"/pbs/"+edge_to_test_on_0+"_"+str(len_)

        if not os.path.exists(problem_folder_name):
            os.makedirs(problem_folder_name) 

            save_stuff(problem_folder_name, edge_to_test_on_0, "init_sym")
            save_stuff(problem_folder_name, edge_to_test_on_1, "goal_sym")
            save_stuff(problem_folder_name, removed_edges, "deleted_edges_sym")

elif args.domain == "sokoban":
    

    if len(to_avoid) > 2:
        print("there are already three problems")
    
    else:


        removed_edges, edge_to_test_on_0, edge_to_test_on_1, len_  = return_transitions_to_remove(18, to_avoid)
        problem_folder_name = args.exp_folder+"/pbs/"+edge_to_test_on_0+"_"+str(len_)
        print("icii")
        print(problem_folder_name)
        if not os.path.exists(problem_folder_name):
            os.makedirs(problem_folder_name)
            save_stuff(problem_folder_name, edge_to_test_on_0, "init_sym")
            save_stuff(problem_folder_name, edge_to_test_on_1, "goal_sym")
            save_stuff(problem_folder_name, removed_edges, "deleted_edges_sym")
