import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
import argparse
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="A script to generate traces from a domain")
parser.add_argument('--trace_dir', type=str, help='trace_dir name')
parser.add_argument('--exp_folder', type=str, help='exp_folder name')
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--remove_some_trans', type=str, required=True)
parser.add_argument('--add_noisy_trans', type=str, required=True)
parser.add_argument('--ten_percent_noisy_and_dupplicated', type=str, required=True)
parser.add_argument('--type_exp', type=str, required=True)
parser.add_argument('--use_transi_iden', type=str, required=True)
parser.add_argument('--pb_folder', type=str, help='problem folder name', required=False)
args = parser.parse_args()

def str_to_bool(s):
    return s.lower() == "true"


    




exp_folder = args.exp_folder
trace_dir = args.trace_dir
remove_some_trans = str_to_bool(args.remove_some_trans)
add_noisy_trans = str_to_bool(args.add_noisy_trans)
ten_percent_noisy_and_dupplicated = str_to_bool(args.ten_percent_noisy_and_dupplicated)
type_exp = args.type_exp
use_transi_iden = None
if args.use_transi_iden == "False":
    use_transi_iden = False
else:
    use_transi_iden = True


cleaness = "noisy"
if add_noisy_trans == False:
    cleaness = "clean"


# erroneous = "faultless"
# if ten_percent_noisy_and_dupplicated == False:
#     erroneous = "erroneous"


dataset_folder_name = exp_folder


def reduce_resolution(image, domain, exp_type):

    reduced_image = None



    if exp_type == "r_latplan":

        if domain == "sokoban":
            # Use np.take to select every 4th element in the first two dimensions
            reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 3), axis=0),
                                    np.arange(0, image.shape[1], 3), axis=1)

        elif domain == "hanoi":
            # Use np.take to select every 4th element in the first two dimensions
            reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                                    np.arange(0, image.shape[1], 9), axis=1)
        elif domain == "blocks":
            # Use np.take to select every 4th element in the first two dimensions
            reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                                    np.arange(0, image.shape[1], 9), axis=1)


    elif exp_type == "vanilla":

        if domain == "sokoban":
            # Use np.take to select every 4th element in the first two dimensions
            reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 6), axis=0),
                                    np.arange(0, image.shape[1], 6), axis=1)

        # SHOULD BE 18  
        elif domain == "blocks":
            reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                                    np.arange(0, image.shape[1], 9), axis=1)

        elif domain == "hanoi":
            # Use np.take to select every 4th element in the first two dimensions
            reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 13), axis=0),
                                    np.arange(0, image.shape[1], 13), axis=1)
    return reduced_image



# # Function to reduce resolution of a single image using np.take
# def reduce_resolution(image):

#     # Use np.take to select every 4th element in the first two dimensions
#     reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 6), axis=0),
#                             np.arange(0, image.shape[1], 6), axis=1)

#     return reduced_image

if args.domain == "hanoi":
    from hanoi_process import normalize, equalize, enhance, preprocess, deenhance, denormalize, unnormalize_colors, normalize_colors


if args.domain == "blocks":
    from blocks_process import normalize, equalize, enhance, preprocess, deenhance, denormalize, unnormalize_colors, normalize_colors


if args.domain == "sokoban":
    from sokoban_process import normalize, equalize, enhance, preprocess, deenhance, denormalize, unnormalize_colors, normalize_colors



#### unpreprocess ===> !!!



def save_dataset(dire, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min):
    data = {
        "train_set": train_set, 
        "test_val_set": test_val_set, 
        "all_pairs_of_images_reduced_orig": all_pairs_of_images_reduced_orig, 
        "all_actions_one_hot": all_actions_one_hot, 
        "mean_all": mean_all, 
        "std_all": std_all, 
        "all_actions_unique": all_actions_unique, 
        "orig_max": orig_max, 
        "orig_min": orig_min
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = "data.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)



def load_pickefile(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data

def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data

import re
# Regular expression to find the pattern [LETTERNUMBER]
pattern = r"\[([a-zA-Z])([0-9])\]"

# Function to replace the pattern [LETTERNUMBER] with LETTERNUMBER
def replace_pattern(match):
    return f"{match.group(1)}{match.group(2)}"

def replace_in_str(word, dom):
        if dom == "hanoi":
            res1 = word.replace(":", "")
            res1 = res1.replace("default", "")
            res1 = res1.replace("peg1", "")
            res1 = res1.replace("peg2", "")
            res1 = res1.replace("peg3", "")
            res1 = res1.replace("peg4", "")
            res1 = res1.replace(" ", "")
            res1 = res1.replace("[]", "E")
            res1 = res1.replace(",", "")
            res1 = transformed_text = re.sub(pattern, replace_pattern, res1)
            return res1

        elif dom == "blocks":
            res1 = word.replace(":block", "")
            res1 = res1.replace("None", "")
            res1 = res1.replace("['", "")
            res1 = res1.replace("']", "")
            return res1

        elif dom == "sokoban":
            res1 = word.replace("5", "")
            res1 = res1.replace("[", "")
            res1 = res1.replace("]", "")
            res1 = res1.replace(" ", "")
            res1 = res1.replace("\n", "")
        

            return [res1[:len(res1)//2], res1[len(res1)//2:]]


        


def save_np_image(dire, array, file_name):
    data = {
        "image": array,
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = str(file_name)+".p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)




# construct pairs (of images)
# and construct the array of action (for each pair of images) 
# Return: pairs of images, corresponding actions (one-hot)
def modify_one_trace(all_images_transfo_tr, all_images_orig_tr, all_actions, all_actions_unique_):

    # building the array of pairs
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    for iiii, p in enumerate(all_images_transfo_tr):
        if iiii%2 == 0:
            #if iiii < len(all_images_transfo_tr)-1:
            all_pairs_of_images.append([all_images_transfo_tr[iiii], all_images_transfo_tr[iiii+1]])
            #all_pairs_of_images_orig.append([all_images_orig_tr[iiii], all_images_orig_tr[iiii+1]])



    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions:
        all_actions_indices.append(all_actions_unique_.index(str(ac)))
    
    actions_indexes = torch.tensor(all_actions_indices).long()


    # array of one hot encoded actions
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique_))






    return all_pairs_of_images, all_pairs_of_images_orig, actions_one_hot.numpy()




def retrieve_files(base_folder):

    dico_pbs = {}

    for root, dirs, files in os.walk(base_folder):
        for subdir in dirs:
            dico_pbs[str(subdir)] = {}
            subdir_path = os.path.join(root, subdir)
            goal_file = os.path.join(subdir_path, 'goal_sym.p')
            init_file = os.path.join(subdir_path, 'init_sym.p')
            deleted_edges = os.path.join(subdir_path, 'deleted_edges_sym.p')
            dico_pbs[str(subdir)]["goal"] = goal_file
            dico_pbs[str(subdir)]["init"] = init_file
            dico_pbs[str(subdir)]["deleted_edges"] = deleted_edges

    return dico_pbs




def create_dico_keySymState_valSubfolder(base_folder):

    dico = {}

    # 
    #
    #   {
    #       'Ed1d2[d4d3]' : 
    #                 [ {'8_0' : 'init'}, {'9_0' : 'init'} etc]
    #                      
    #                       
    #                       
    #                   
    #   }

    for root, dirs, files in os.walk(base_folder):
        for subdir in dirs:
            

            subdir_path = os.path.join(root, subdir)
            goal_file = os.path.join(subdir_path, 'goal_sym.p')
            init_file = os.path.join(subdir_path, 'init_sym.p')
            deleted_edges = os.path.join(subdir_path, 'deleted_edges_sym.p')

 
            # print(load_pickefile(init_file))
            # exit()

            if os.path.exists(goal_file):
                loaded_goal = load_pickefile(goal_file)["stuff"]
                if loaded_goal not in dico:
                    dico[loaded_goal] = []

                dico[loaded_goal].append({ str(subdir): 'goal'})



            if os.path.exists(init_file):
                loaded_init = load_pickefile(init_file)["stuff"]
                print(loaded_init)
                print(dico)
                print(type(dico))
                if loaded_init not in dico:
                    dico[loaded_init] = []

                dico[loaded_init].append({ str(subdir): 'init'})

            # if os.path.exists(deleted_edges):
            #     loaded_deleted_edges = load_pickefile(deleted_edges)
            #     if loaded_deleted_edges["stuff"] not in dico:
            #         dico[loaded_deleted_edges["stuff"]] = []

            # dico[str(subdir)]["goal"] = goal_file
            # dico[str(subdir)]["init"] = init_file
            # dico[str(subdir)]["deleted_edges"] = deleted_edges

    return dico



def return_trans_parts(transitioo, domain_):
    

    ## stringify both states
    if domain_ == "hanoi":
        first_split_ = transitioo[1][3].split("}, {")
    elif domain_ == "blocks":
        first_split_ = transitioo[1].split("', '")


    if domain_ != "sokoban":

        part1_trans_ = replace_in_str(first_split_[0].replace('[{', ''), domain_)
        part2_trans_ = replace_in_str(first_split_[1].replace('}]', ''), domain_)

    else:
        part1_trans_ = replace_in_str(transitioo[1], "sokoban")[0]
        part2_trans_ = replace_in_str(transitioo[1], "sokoban")[1]

    return part1_trans_, part2_trans_


def compute_distance(image1, image2):
    """
    Computes the Euclidean distance between two images.
    
    Parameters:
    image1 (numpy array): The first image array of shape (45, 45, 3)
    image2 (numpy array): The second image array of shape (45, 45, 3)
    
    Returns:
    float: The Euclidean distance between the two images.
    """
    # Ensure both images are numpy arrays
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    
    # Check if the shapes match
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same shape.")
    
    # Flatten the images to 1D arrays
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # Compute the Euclidean distance
    distance = np.linalg.norm(image1_flat - image2_flat)
    
    return distance


def transition_identifier(image1, image2, dico_trans_, domain):

    threshold = None
    if domain == "sokoban":
        threshold = 20
    elif domain == "hanoi":
        threshold = 5
    elif domain == "blocks":
        threshold = 20
    #print("len dico_trans_ {}".format(len(dico_trans_)))

    trans_already_exist_and_found = False

    for id_, images in dico_trans_.items():

        # first image
        #images[1] # 

        dist_pres = compute_distance(image1, images[0])


        print("dist_pres {}".format(str(dist_pres)))
        if dist_pres < threshold:

            dist_sucs = compute_distance(image2, images[1])

            if dist_sucs < threshold:
                trans_already_exist_and_found = True

                return id_, trans_already_exist_and_found

    # 
    return len(dico_trans_)+1, trans_already_exist_and_found




## ALL EXPS:
##      Exp1: complete clean faultless   (1 training per domain)        DONE
##      Exp2: complete noisy faultless   (1 training per domain)
##      Exp3: partial clean faultless   
##      Exp4: partial noisy faultless
##      Exp5: complete noisy erroneous ("symbols" are same as Exp2)
##
##      3 longest, 3 middle, 3 quarters, 1 steps

def process_dataset(
        domain="hanoi",
        type_exp = "r_latplan",
        build_dfa=False,
        number_noisy_versions=3,
        create_init_goal=True,
        perc=55,
        std=0.015, #0.015
        ):

    #   domain
    #   type_exp (pour le type de reduction) DONE
    #   number_noisy_versions (dès que dataset est noisy)
    #   perc (dès que dataset est noisy)
    #   std  (dès que dataset est noisy)
    #   use_transi_iden (si on utilise ou pas le transi identifier)   


    #  AUTRE DONNEES
    # exp_folder
    # trace_dir
    # remove_some_trans
    # add_noisy_trans
    # ten_percent_noisy_and_dupplicated
    # type_exp
    # cleaness = "noisy"
    # if add_noisy_trans == False:
    #     cleaness = "clean"
    # dataset_folder_name = exp_folder


    # CHECK QUE TOUT CONCORDE POUR CHAQUE SCENARIO

    # EXP1 (complete clean faultless)
    # EXP2 
    # ....





    # Folder where all the pbs of the exp are stored
    problems_folder = exp_folder + "/pbs/"


    #dico_pbs_and_deleted_edges = retrieve_files(problems_folder) !!! delete ? !!!!!

    # Dico with keys = symbolic states present in the problems, values = name of the pbs folder (e.g '8_0)
    dico_keySymState_valSubfolder = create_dico_keySymState_valSubfolder(problems_folder)


    # pour les deleted edges ?

    # some variables
    orig_max = None
    orig_min = None
    mean = 0

    loaded_dataset = load_dataset(trace_dir+"/traces.p")

    ## Arrays and Dicos

    all_images_reduced = [] # raw reduced images listed by the traces
    all_images_reduced_and_norm = [] # raw reduced images and normed, listed by the traces

    all_images_reduced_and_norm_uniques_nonnoisy = {} # dico with non noisy images only
    all_images_reduced_and_norm_uniques_noisy = {} # dico with noisy images only

    all_pairs_of_images_reduced_orig = [] # pairs of raw reduced images listed by the traces
    all_actions_one_hot = [] # all the actions (one-hot) listed by the traces

    traces_indices = [] # for each trace, the indices of where it starts/ends in the total gathering of traces (e.g. in all_pairs_of_images_reduced_orig)

    start_trace_index = 0
    all_actions_for_trace = [] # for each trace, an array of all the actions (given as pairs of str)
    all_transitions_unique = [] # all the transitions, listed by the traces, each item = two image arrays + the two str representing each state
    all_the_unique_actions_noisy_part = [] # all the actions labels (with both images being noisy)
    all_the_unique_actions = [] # all the actions labels
    all_the_unique_actions_noisy_part_count = {} # dico to count the  # of noisy version per action
    all_the_unique_actions_ten_percent_noisy_and_dup = [] # list of actions label of the group 'noisy and dup'
    all_actions_unique = []



    # COMPUTING THE WHOLE DATASET mean and std
    unique_obs_img = loaded_dataset["unique_obs_img"]
    reduced_uniques = []
    for uniq in unique_obs_img:
        reduced_uniques.append(reduce_resolution(uniq, domain, type_exp))
        reduced_uniques.append(reduce_resolution(uniq, domain, type_exp))
        reduced_uniques.append(reduce_resolution(uniq, domain, type_exp))



    if domain  != "sokoban":
        unique_obs_img_preproc, orig_max, orig_min = preprocess(reduced_uniques, histo_bins=24)
        unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)
    else:
        unique_obs_img_color_norm, mean_all, std_all = normalize_colors(reduced_uniques, mean=None, std=None)


    # avant de faire la loop qui crée les images, possible de
    # fixer le array des actions uniques ? 
    #
    #   même avec le transition identifier ?
    #
    #          Oui possible ET PLUS CLAIR

    #
    #    1) parcours des traces
    #
    #    2) à chaque paire, création du label (en fonction du scenar)

    ##      Exp1: complete clean faultless  (with / without TI)
    ##      Exp2: complete noisy faultless  (with / without TI)
    ##      Exp3: partial clean faultless   (with / without TI)
    ##      Exp4: partial noisy faultless    (with / without TI)
    ##      Exp5: complete noisy erroneous ("symbols" are same as Exp2)
    #          

    #   3) if without TI

    #           Exp1: avec transi[1] , part1_trans_ etc
    #           Exp2: idem
    #           Exp3: idem (car loaded_traces est MAJ avant le parcours)
    #           Exp4: idem
    #           Exp5: i) création du groupe de 10% de labels (parmis les 55% noisy) ii) puis
    #                       ajout au vecteur des labels d'actions (avec +1 -> +3, et enlever le '+0')
    
    #   4) if WITH TI
    #   
    #           Exp1: à chaque paire, appel de la fonction, si appartient à dico_trans etc
    #                   BIEN vérifier que c'est le même compte que (WITH TI) (sinon, voir quelles sont les diffs)
    #           Exp2: idem (même compte que Exp1, en être sûr !!!!)
    #           Exp3: idem compte que Without et Exp3
    #           Exp4: idem compte que Without et Exp4
    #           Exp5: weird case.... savoir en avance quels IDS (c'est des entiers là), on veut duppliquer
    #                               et lors de loop avec creation d'images, alors, créer aussi un ID supplémentaire
    #                   

    loaded_traces = loaded_dataset["traces"]
    print("exp_folderexp_folder")
    print(exp_folder)


    traces_for_each_partial = []
    # maj the traces (only keep the ones specified in the dfa)
    if remove_some_trans:

        


        for root, dirs, files in os.walk(exp_folder+"/pbs"):
            print(dirs)
        

            for subdir in dirs:

                # print(subdir)
                # print(args.pb_folder)
                # exit()
                if subdir != args.pb_folder:
                    continue


                transis_to_remove = []
                # print("ggggggg")
                # print(exp_folder+"/pbs/"+subdir+"/deleted_edges_sym.p")
                # exit()
                loaded_removable_trans = load_dataset(exp_folder+"/pbs/"+subdir+"/deleted_edges_sym.p")
                transis_to_remove = loaded_removable_trans["stuff"]

                loaded_removable_edge_goal = load_dataset(exp_folder+"/pbs/"+subdir+"/goal_sym.p")
                edge_removed_goal = loaded_removable_edge_goal["stuff"]
                goal_st = edge_removed_goal

                loaded_removable_edge_init = load_dataset(exp_folder+"/pbs/"+subdir+"/init_sym.p")
                edge_removed_init = loaded_removable_edge_init["stuff"]
                init_st = edge_removed_init
                


                transis_to_remove_str = []
                for tr in transis_to_remove:
                    transis_to_remove_str.append(str(tr))

                loaded_traces_after_prunning = []

                for iii, trace in enumerate(loaded_traces):
                    #trace_copy = trace.copy()
                    if len(trace) == 0:
                        continue
                    
                    trace_tmp = []
                    for trtrtr, transitio in enumerate(trace):


                        part1, part2 = return_trans_parts(transitio, domain)

                        if str([part1, part2]) not in transis_to_remove_str:
                            trace_tmp.append(transitio)

                    #print("len trace_tmp {}".format(str(len(trace_tmp))))

                    loaded_traces_after_prunning.append(trace_tmp)       

                loaded_traces = loaded_traces_after_prunning

                break

            break

    # all_actions_unique

    # build the vector of unique action's labels
    # size depends on domain and if complete/not
    
    dico_trans = {}
    for iii, trace in enumerate(loaded_traces):
        for trtrtr, transitio in enumerate(trace):
            if use_transi_iden:
                if domain != "sokoban":
                    transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0], domain, type_exp), histo_bins=24)
                    transi_0_reduced_and_norm, _, _ = normalize_colors(transi_0_prepro, mean=mean_all, std=std_all)

                    transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1], domain, type_exp), histo_bins=24)
                    transi_1_reduced_and_norm, _, _ = normalize_colors(transi_1_prepro, mean=mean_all, std=std_all)
                else:
                    transi_0_reduced_and_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][0], domain, type_exp), mean=mean_all, std=std_all)
                    transi_1_reduced_and_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][1], domain, type_exp), mean=mean_all, std=std_all)

                # print(compute_distance(transi_0_reduced_and_norm, transi_1_reduced_and_norm)) 
                # print(compute_distance(transi_1_reduced_and_norm, transi_1_reduced_and_norm))
                # exit()
                idd, trans_exist = transition_identifier(transi_0_reduced_and_norm, transi_1_reduced_and_norm, dico_trans, domain)
                if not trans_exist:
                    all_the_unique_actions.append(str(idd))
                    dico_trans[idd] = [transi_0_reduced_and_norm, transi_1_reduced_and_norm]
            else:
                part1, part2 = return_trans_parts(transitio, domain)

                if str([part1, part2]) not in all_the_unique_actions:
                    all_the_unique_actions.append(str([part1, part2]))


    # plusieurs tableaux / dicos
    #   
    #       all_the_unique_actions ..
    #       
    #       all_the_unique_actions_noisy_part
    #
    #       all_the_unique_actions_ten_percent_noisy_and_dup

    # sans transi identifier
    #
    #   parcours des traces, 
    #
    #       identification du label de la transition
    #
    #           EN  FCT de son appartenance à un des 3 tableaux ci dessous
    #                       mettre soit une version non noisy, soit noisy (mais même action label)
    #                               soit une version où tout est noisy (dont action label)
    #
    #                   

    # avec transi identifier

    # create two arrays to specify the subgroups of noisy transitions and duplicated ones
    # (all_the_unique_actions_noisy_part) and (all_the_unique_actions_ten_percent_noisy_and_dup)


    # if there are noisy transition in the dataset
    if add_noisy_trans:

        random.shuffle(all_the_unique_actions)

        # an array of 55% of the action's labels (<=> the noisy transitions)
        all_the_unique_actions_noisy_part = all_the_unique_actions[:int(len(all_the_unique_actions) * perc / 100)]

        all_the_unique_actions_noisy_part_count = {key: 0 for key in all_the_unique_actions_noisy_part}


        if ten_percent_noisy_and_dupplicated:
            all_the_unique_actions_ten_percent_noisy_and_dup = all_the_unique_actions_noisy_part[:int(len(all_the_unique_actions_noisy_part) * 1.8 / 100)]


        all_the_unique_actions_noisy_part_count = {key: 0 for key in all_the_unique_actions_noisy_part}

        print("Number of UNIQUE noisy transs {}".format(str(len(all_the_unique_actions_noisy_part_count)))) # 81



    print("all_the_unique_actions {}".format(str(len(all_the_unique_actions))))
    print("all_the_unique_actions_noisy_part_count {}".format(str(len(all_the_unique_actions_noisy_part_count))))
    print("all_the_unique_actions_ten_percent_noisy_and_dup {}".format(str(len(all_the_unique_actions_ten_percent_noisy_and_dup))))
    
    #return None,None,None,None,None,None,None,None,None

    ###################### COMPUTING THE WHOLE DATASET mean and std #####################
    unique_obs_img = loaded_dataset["unique_obs_img"]
    # array used for computing the whole mean / std all the images (3 copies of each image)
    reduced_uniques = []
    for uniq in unique_obs_img:
        # print(type(uniq))
        # print(uniq.shape)
        # if reduce_resolution(uniq, domain, type_exp) is None:
        #     print("was none")
        #     exit()
        reduced_uniques.append(reduce_resolution(uniq, domain, type_exp))
        reduced_uniques.append(reduce_resolution(uniq, domain, type_exp))
        reduced_uniques.append(reduce_resolution(uniq, domain, type_exp))


    if domain  != "sokoban":
        unique_obs_img_preproc, orig_max, orig_min = preprocess(reduced_uniques, histo_bins=24)
        unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)
    else:
        unique_obs_img_color_norm, mean_all, std_all = normalize_colors(reduced_uniques, mean=None, std=None)





    def save_init_or_goal_image(dico_keySymState_valSubfolder_, partx_trans):
        for dicc in dico_keySymState_valSubfolder_[partx_trans]:
            rep1 = list(dicc.keys())[0]
            mode1 = list(dicc.values())[0]
            save_np_image(problems_folder+"/"+rep1, transi_0_reduced_and_norm, mode1)
            if domain != "sokoban":
                unorm = unnormalize_colors(transi_0_reduced_and_norm, mean_all, std_all) 
                dehanced = deenhance(unorm)
                denormalized = denormalize(dehanced, orig_min, orig_max)
                denormalized = np.clip(denormalized, 0, 1)
            else:
                unorm = unnormalize_colors(transi_0_reduced_and_norm, mean_all, std_all) 
                denormalized = np.clip(unorm, 0, 1)     
            plt.imsave(problems_folder+"/"+rep1+"/"+partx_trans+"_"+mode1+".png", denormalized)
            plt.close()
        return




    longest_plan_transis_test = []

    test_noisy_transis = []

    already_created = False


    ###################### MAIN LOOP #####################
    #
    #
    #   fait pour quoi cette loop ?
    #   
    #           looper over ttes les traces
    #
    #                   et ajouter les actions (dans ) et la pairs d'images (dans all_images_reduced_and_norm)
    #
    #
    #                   Aussi création des init/goal images ET des indices



    # loop over traces
    for iii, trace in enumerate(loaded_traces):


        nb_additional_noisy_trans = 0

        actions_for_one_trace = []


        # loop over transitions of the trace
        for trtrtr, transitio in enumerate(trace):

            already_created = False


            # normalize the two images
            if domain != "sokoban":
                transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0], domain, type_exp), histo_bins=24)
                transi_0_reduced_and_norm, _, _ = normalize_colors(transi_0_prepro, mean=mean_all, std=std_all)

                transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1], domain, type_exp), histo_bins=24)
                transi_1_reduced_and_norm, _, _ = normalize_colors(transi_1_prepro, mean=mean_all, std=std_all)
            else:
                transi_0_reduced_and_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][0], domain, type_exp), mean=mean_all, std=std_all)
                transi_1_reduced_and_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][1], domain, type_exp), mean=mean_all, std=std_all)


            ## stringify both states (used for the Without TI case)
            if domain == "hanoi":
                first_split_ = transitio[1][3].split("}, {")
            elif domain == "blocks":
                first_split_ = transitio[1].split("', '")

            if domain != "sokoban":
                part1_trans_ = replace_in_str(first_split_[0].replace('[{', ''), domain)
                part2_trans_ = replace_in_str(first_split_[1].replace('}]', ''), domain)

            else:
                part1_trans_ = replace_in_str(transitio[1], "sokoban")[0]
                part2_trans_ = replace_in_str(transitio[1], "sokoban")[1]



            # identify the "canonical" identifier for the transition
            canon_id = None
            if use_transi_iden:
                canon_id, trans_exist = transition_identifier(transi_0_reduced_and_norm, transi_1_reduced_and_norm, dico_trans, domain)
            else:
                canon_id = str([part1_trans_, part2_trans_])






            #### SAVING THE INIT/GOAL IMAGES (non noisy version)
            if cleaness == "clean" and create_init_goal and not already_created:

                # if the first image is a key in dico_keySymState_valSubfolder
                if part1_trans_ in dico_keySymState_valSubfolder.keys():

                    # go over the corresponding values (which repr. folder_name / type (btween init and goal))
                    save_init_or_goal_image(dico_keySymState_valSubfolder, part1_trans_)
                
                if part2_trans_ in dico_keySymState_valSubfolder.keys():
                    save_init_or_goal_image(dico_keySymState_valSubfolder, part2_trans_)

                if canon_id not in all_actions_unique:
                    all_actions_unique.append(canon_id)

            # If the current transition is NOT NOISY, add the the pair and the action label
            if cleaness == "clean": # and str([part1_trans_, part2_trans_]) not in all_the_unique_actions_noisy_part_count:

                # add each image to "all_images_reduced_and_norm"
                all_images_reduced_and_norm.append(transi_0_reduced_and_norm)
                all_images_reduced_and_norm.append(transi_1_reduced_and_norm)
                
                # add the transition's label to "actions_for_one_trace"
                actions_for_one_trace.append(canon_id)

            # elif we are in a "noisy" experiment
            elif cleaness == "noisy":
            

                # check if the current transitions should be noised
                if canon_id in all_the_unique_actions_noisy_part_count:


                    # update the count (DELETE ??????)
                    #all_the_unique_actions_noisy_part_count[str([part1_trans_, part2_trans_])] += 1
                    

                    # create the different noisy versions
                    for inndex in range(number_noisy_versions):

                        nb_additional_noisy_trans += 1

                        # create the noised images
                        if domain != "sokoban":
                            transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0], domain, type_exp), histo_bins=24)
                            transi_0_prepro_img_color_norm, _, _ = normalize_colors(transi_0_prepro, mean=mean_all, std=std_all)
                        else:
                            transi_0_prepro_img_color_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][0], domain, type_exp), mean=mean_all, std=std_all)
                        random.seed(inndex)
                        np.random.seed(inndex)
                        gaussian_noise_0 = np.random.normal(mean, std, transi_0_prepro.shape)
                        random.seed(1)
                        np.random.seed(1)
                        noisy1_reduced_preproc_norm = transi_0_prepro_img_color_norm + gaussian_noise_0
                        noisy1 = noisy1_reduced_preproc_norm

                        if domain != "sokoban":
                            transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1], domain, type_exp), histo_bins=24)
                            transi_1_prepro_img_color_norm, _, _ = normalize_colors(transi_1_prepro, mean=mean_all, std=std_all)
                        else:
                            transi_1_prepro_img_color_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][1], domain, type_exp), mean=mean_all, std=std_all)
                        random.seed(inndex)
                        np.random.seed(inndex)
                        gaussian_noise_1 = np.random.normal(mean, std, transi_1_prepro.shape)
                        random.seed(1)
                        np.random.seed(1)
                        noisy2_reduced_preproc_norm = transi_1_prepro_img_color_norm + gaussian_noise_1
                        noisy2 = noisy2_reduced_preproc_norm

                        # immmm = transitio[0][0]
                        # np.random.seed(55)
                        # gaussian_noise_1 = np.random.normal(mean, 20, immmm.shape)
                        # immmm = immmm + gaussian_noise_1
                        # immmm = np.clip(immmm, 0, 255).astype(np.uint8) 
                        # plt.imsave("HANOI_NOISY1.png", immmm)
                        # plt.close()


                        # # add the two images to "all_images_reduced_and_norm_uniques_noisy" (for generating the PDDL state for exp5)
                        # if str(part1_trans_) not in all_images_reduced_and_norm_uniques_noisy:
                        #     all_images_reduced_and_norm_uniques_noisy[str(part1_trans_)] = noisy1_reduced_preproc_norm

                        # if str(part2_trans_) not in all_images_reduced_and_norm_uniques_noisy:
                        #     all_images_reduced_and_norm_uniques_noisy[str(part1_trans_)] = noisy2_reduced_preproc_norm


                        # if 'noise' init/state was chosen
                        # AND if one of the present states is init/state
                        # save the colored normalized array in some picke file
                        if create_init_goal:


                            if part1_trans_ in dico_keySymState_valSubfolder.keys():
                                for dicc in dico_keySymState_valSubfolder[part1_trans_]:
                                    rep3 = list(dicc.keys())[0]
                                    mode3 = list(dicc.values())[0]
                                    save_np_image(problems_folder+"/"+rep3, noisy1_reduced_preproc_norm, mode3+"_"+str(inndex))
                                    
                                    unorm = unnormalize_colors(noisy1_reduced_preproc_norm, mean_all, std_all) 
                                    dehanced = deenhance(unorm)
                                    denormalized = denormalize(dehanced, orig_min, orig_max)
                                    denormalized = np.clip(denormalized, 0, 1) 


                                    plt.imsave(problems_folder+"/"+rep3+"/"+part1_trans_+"_"+mode3+"_"+str(inndex)+".png", denormalized)
                                    plt.close()


                            if part2_trans_ in dico_keySymState_valSubfolder.keys():
                                for dicc in dico_keySymState_valSubfolder[part2_trans_]:
                                    rep4 = list(dicc.keys())[0]
                                    mode4 = list(dicc.values())[0]
                                    print("rep4 {}".format(rep4))
                                    print("mode4 {}".format(mode4))
                                    #exit()
                                    save_np_image(problems_folder+"/"+rep4, noisy2_reduced_preproc_norm, mode4+"_"+str(inndex))
                                    
                                    unorm = unnormalize_colors(noisy2_reduced_preproc_norm, mean_all, std_all) 
                                    dehanced = deenhance(unorm)
                                    denormalized = denormalize(dehanced, orig_min, orig_max)
                                    denormalized = np.clip(denormalized, 0, 1)   
                                    
                                    plt.imsave(problems_folder+"/"+rep4+"/"+part2_trans_+"_"+mode4+"_"+str(inndex)+".png", denormalized)
                                    plt.close()

                        # add the noisy images to "all_images_reduced_and_norm" 
                        all_images_reduced_and_norm.append(noisy1_reduced_preproc_norm)
                        all_images_reduced_and_norm.append(noisy2_reduced_preproc_norm)
                        print("ici22")
                        
                        # If the transitions labels must be differentiated !
                        if canon_id in all_the_unique_actions_ten_percent_noisy_and_dup:

                            # add the label of the transition to "actions_for_one_trace"
                            actions_for_one_trace.append(canon_id+"_"+str(inndex))
                            # add the label of the transition to all_actions_unique and to "all_transitions_unique" 
                            if canon_id+"_"+str(inndex) not in all_actions_unique:

                                all_actions_unique.append(canon_id+"_"+str(inndex))
                                #all_transitions_unique.append([part1_trans_+"_"+str(inndex), part2_trans_+"_"+str(inndex)])
                        
                        else:
                            actions_for_one_trace.append(canon_id)

                            # add the action to the unique vectors
                            if canon_id not in all_actions_unique:
                                all_actions_unique.append(canon_id)
                                #all_transitions_unique.append([part1_trans_, part2_trans_])
                




        # add the actions of this trace to all_actions_for_trace (the array of all actions of all traces)
        all_actions_for_trace.append(actions_for_one_trace)

   
        # add the indices for this trace to traces_indices array
        traces_indices.append([start_trace_index, start_trace_index+(len(trace)+nb_additional_noisy_trans)*2])
        
        # update the index for the next trace
        start_trace_index+=(len(trace)+nb_additional_noisy_trans)*2


    # for k, v in all_images_reduced_and_norm_uniques_noisy.items():
    #     # print("SHOULD NOT BE HERE")
    #     # exit()
    #     save_np_image(exp_folder_all_states, v, "image"+str(k)+"noisy") 



    if build_dfa:

        ############################################
        ######## Building the file for the DFA #####
        ############################################

        # Input:
        # all_actions_unique




        # WE NEED
        # initial_state
        # final_state
        # alphabet_dic (for each transition (key) we associate a name ("a"+str(i)))
        # unique_states
        # all_transitions where each item is the transi described as firstState+nameAction+SecondState

        total_dfa_dico = {}
        initial_state = all_transitions_unique[0][0]
        final_state = all_transitions_unique[-1][1]

        alphabet_dic = {}
        unique_states = []
        all_transitions_dfa = []

        # init_st = "Ed4E[d3d2d1]"
        # goal_st = "EEE[d4d3d2d1]"

        # print("lla")
        # print(all_transitions_unique[0])
        # exit()

        for iu, tran in enumerate(all_transitions_unique):

            alphabet_dic[str(tran)] = "a"+str(iu)

            if tran[0] not in unique_states:
                unique_states.append(str(tran[0]))

            if tran[1] not in unique_states:
                unique_states.append(str(tran[1]))


        for iu, tran in enumerate(all_transitions_unique):
            transi = []
            transi.append(str(tran[0]))
            transi.append(alphabet_dic[str(tran)])
            transi.append(str(tran[1]))
            all_transitions_dfa.append(transi)


        total_dfa_dico["alphabet"] = list(alphabet_dic.values())
        total_dfa_dico["states"] = unique_states
        total_dfa_dico["initial_state"] = initial_state
        #total_dfa_dico["initial_state"] = ""
        total_dfa_dico["accepting_states"] = final_state
        #total_dfa_dico["accepting_states"] = "" # final_state
        total_dfa_dico["transitions"] = all_transitions_dfa

        with open("DFA_Hanoi_4_4.json", 'w') as f:
            json.dump(total_dfa_dico, f)


        ############################################
        #### END Building the file for the DFA #####
        ############################################


    with open("AAAAll_actionsPRE.txt","w") as f:
    
        for i in range(len(all_the_unique_actions)):
            #np.argmax(all_actions_one_hot[i]
            f.write("a"+str(i)+" is "+str(all_the_unique_actions[i])+"\n")


    all_images_reduced_gaussian_20_norm = all_images_reduced_and_norm
    all_images_reduced_gaussian_30_norm = all_images_reduced_and_norm
    all_images_reduced_gaussian_40_norm = all_images_reduced_and_norm


    all_pairs_of_images_processed_gaussian20 = []
    all_pairs_of_images_processed_gaussian30 = []
    all_pairs_of_images_processed_gaussian40 = []




    # second loop to process the pairs
    for iiii, trace in enumerate(loaded_traces):


        all_images_transfo_tr_gaussian20 = all_images_reduced_gaussian_20_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]
        all_images_orig_reduced_tr = all_images_reduced[traces_indices[iiii][0]:traces_indices[iiii][1]]

        all_images_transfo_tr_gaussian30 = all_images_reduced_gaussian_30_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]
        all_images_transfo_tr_gaussian40 = all_images_reduced_gaussian_40_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]


        all_actions_tr = all_actions_for_trace[iiii]
        

        # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace
        all_pairs_of_images_of_trace_gaussian20, all_pairs_of_images_orig_reduced_of_trace, actions_one_hot_of_trace = modify_one_trace(all_images_transfo_tr_gaussian20, all_images_orig_reduced_tr, all_actions_tr, all_the_unique_actions)
        all_pairs_of_images_of_trace_gaussian30, _, _ = modify_one_trace(all_images_transfo_tr_gaussian30, all_images_orig_reduced_tr, all_actions_tr, all_the_unique_actions)
        all_pairs_of_images_of_trace_gaussian40, _, _ = modify_one_trace(all_images_transfo_tr_gaussian40, all_images_orig_reduced_tr, all_actions_tr, all_the_unique_actions)

        all_pairs_of_images_processed_gaussian20.extend(all_pairs_of_images_of_trace_gaussian20)
        all_pairs_of_images_processed_gaussian30.extend(all_pairs_of_images_of_trace_gaussian30)
        all_pairs_of_images_processed_gaussian40.extend(all_pairs_of_images_of_trace_gaussian40)
        
        #all_pairs_of_images.extend(all_pairs_of_images_of_trace)
        all_pairs_of_images_reduced_orig.extend(all_pairs_of_images_orig_reduced_of_trace)
        all_actions_one_hot.extend(actions_one_hot_of_trace)





    ###   alternatively take 2 copies from 20/30, 20/40, 30/40 and put them in the training set
    ###   and 1 copy for the other gaussian array (resp. 40 30 20) and put it in the test_val set

    train_set = []
    test_val_set = []
    for i in range(0, len(all_pairs_of_images_processed_gaussian20)):
        if i%3 == 0:
            train_set.append([all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i]])
            train_set.append([all_pairs_of_images_processed_gaussian30[i], all_actions_one_hot[i]])
            test_val_set.append([all_pairs_of_images_processed_gaussian40[i], all_actions_one_hot[i]])
        elif (i+1)%3 == 0:
            train_set.append([all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i]])
            train_set.append([all_pairs_of_images_processed_gaussian40[i], all_actions_one_hot[i]])
            test_val_set.append([all_pairs_of_images_processed_gaussian30[i], all_actions_one_hot[i]])
        elif (i+2)%3 == 0:
            train_set.append([all_pairs_of_images_processed_gaussian30[i], all_actions_one_hot[i]])
            train_set.append([all_pairs_of_images_processed_gaussian40[i], all_actions_one_hot[i]])
            test_val_set.append([all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i]])


    return train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min



# if partial, the folder is changed outside

# 


if args.domain == "hanoi":
    train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = process_dataset(domain="hanoi", type_exp=type_exp)
    save_dataset(dataset_folder_name, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min)

elif args.domain == "blocks":
    train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = process_dataset(domain="blocks", type_exp=type_exp)
    save_dataset(dataset_folder_name, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min)

elif args.domain == "sokoban":
    train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = process_dataset(domain="sokoban", type_exp=type_exp)
    save_dataset(dataset_folder_name, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min)