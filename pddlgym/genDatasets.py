import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
import argparse
import random


parser = argparse.ArgumentParser(description="A script to generate traces from a domain")
parser.add_argument('--trace_dir', type=str, help='trace_dir name')
parser.add_argument('--exp_folder', type=str, help='exp_folder name')
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--remove_some_trans', type=str, required=True)
parser.add_argument('--add_noisy_trans', type=str, required=True)
parser.add_argument('--ten_percent_noisy_and_dupplicated', type=str, required=True)
args = parser.parse_args()
def str_to_bool(s):
    return s.lower() == "true"
    
exp_folder = args.exp_folder
trace_dir = args.trace_dir
remove_some_trans = str_to_bool(args.remove_some_trans)
add_noisy_trans = str_to_bool(args.add_noisy_trans)
ten_percent_noisy_and_dupplicated = str_to_bool(args.ten_percent_noisy_and_dupplicated)

completness = "partial"
if remove_some_trans == "False":
    completness = "complete"

cleaness = "noisy"
if add_noisy_trans == "False":
    cleaness = "clean"


erroneous = "faultless"
if ten_percent_noisy_and_dupplicated == "False":
    erroneous = "erroneous"


dataset_folder_name = exp_folder


# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):

    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                            np.arange(0, image.shape[1], 9), axis=1)

    return reduced_image

if args.domain == "hanoi":
    from hanoi_process import normalize, equalize, enhance, preprocess, deenhance, denormalize, unnormalize_colors, normalize_colors


if args.domain == "blocks":
    from blocks_process import normalize, equalize, enhance, preprocess, deenhance, denormalize, unnormalize_colors, normalize_colors


if args.domain == "sokoban":
    from sokoban_process import normalize, equalize, enhance, preprocess, deenhance, denormalize, unnormalize_colors, normalize_colors





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



def process_dataset(
        domain="hanoi",
        action_type="full",
        build_dfa=False,
        noise_type='preprocess',
        number_noisy_versions=3,
        init_st = "[d4d2]d1d3E",
        goal_st = "d4d1d3d2",
        create_init_goal=True,
        perc=55,
        std=0.015    #0.015
        ):





    print(exp_folder)
    print(trace_dir)
    print(remove_some_trans)
    print(add_noisy_trans)
    print(ten_percent_noisy_and_dupplicated)
    print(domain)
    print("domain")
    #exit()
    # WAY TO MANY FOLDERS...
    #  WHAT FOLDERS TO BE CREATED TO YOU NEED ? 
    #    Let's say none ? 

    # besoin de 




    global counntteerr

    trans_to_remove_list = ['[d4d2]d1d3E', 'd4d1d3d2']


    # mean and std for the noise
    mean = 0

    loaded_dataset = load_dataset(trace_dir+"/traces.p")

    all_images_reduced = [] # all the raw images in the order listed by the traces
    all_images_reduced_and_norm = []

    all_images_reduced_and_norm_uniques_nonnoisy = {}
    all_images_reduced_and_norm_uniques_noisy = {}

    all_pairs_of_images_reduced_orig = [] # all the pairs of raw images in the order of the traces
    all_actions_one_hot = [] # all the actions (one-hot) in the order listed by the traces
    all_actions_unique = [] # all the actions (pairs of str) in the order listed by the traces

    traces_indices = [] # contains, for each trace, the indices of where it starts / ends in the total gathering of traces (e.g. in all_pairs_of_images_reduced_orig)

    start_trace_index = 0
    all_actions_for_trace = [] # for each trace, an array of all the actions (given as pairs of str)

    all_transitions_unique = [] # all the transitions, in the order...., each item has the two image arrays and the two str representing each state

    all_the_unique_actions_noisy_part = []

    all_the_unique_actions = []
    all_the_unique_actions_noisy_part_count = {}
    all_the_unique_actions_ten_percent_noisy_and_dup = []

    loaded_traces = loaded_dataset["traces"]

    

    if remove_some_trans:

        transis_to_remove = []
        if remove_some_trans:
            loaded_removable_trans = load_dataset(exp_folder+"/deleted_edges.p")
            transis_to_remove = loaded_removable_trans["stuff"]

            loaded_removable_edge_goal = load_dataset(exp_folder+"/goal_state.p")
            edge_removed_goal = loaded_removable_edge_goal["stuff"]
            goal_st = edge_removed_goal

            loaded_removable_edge_init = load_dataset(exp_folder+"/init_state.p")
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

                if domain == "hanoi":
                    first_split_ = transitio[1][3].split("}, {")
                elif domain == "blocks":
                    first_split_ = transitio[1].split("', '")

                part1_trans_ = replace_in_str(first_split_[0].replace('[{', ''), domain)
                part2_trans_ = replace_in_str(first_split_[1].replace('}]', ''), domain)

                if str([part1_trans_, part2_trans_]) not in transis_to_remove_str:
                    trace_tmp.append(transitio)

            #print("len trace_tmp {}".format(str(len(trace_tmp))))

            loaded_traces_after_prunning.append(trace_tmp)       

        loaded_traces = loaded_traces_after_prunning


    if add_noisy_trans:

        random.shuffle(all_the_unique_actions)

        #print(int(len(all_the_unique_actions) * perc / 100))

        #all_the_unique_actions_noisy_part = all_the_unique_actions[:len(all_the_unique_actions)//int(100/perc)]
        all_the_unique_actions_noisy_part = all_the_unique_actions[:int(len(all_the_unique_actions) * perc / 100)]

        #print("len all_the_unique_actions_noisy_part 0") # 798

        longest_plans = [
            ['[d3d2d1]Ed4E', '[d3d2]Ed4d1', 'd3E[d4d2]d1', 'Ed3[d4d2]d1', 'E[d3d2]d4d1', 'd4[d3d2]Ed1', 'd4d3d2d1', '[d4d3]Ed2d1', '[d4d3d2]EEd1', '[d4d3d2d1]EEE'],
            ['Ed3E[d4d2d1]', 'd1d3E[d4d2]', 'd1Ed3[d4d2]', 'd1E[d3d2]d4', 'd1d4[d3d2]E', 'd1d4d3d2', 'd1[d4d3]Ed2', 'd1[d4d3d2]EE', 'E[d4d3d2d1]EE'],
            ['Ed3E[d4d2d1]', 'Ed3d1[d4d2]', 'E[d3d2]d1d4', 'd4[d3d2]d1E', '[d4d2]d3d1E', '[d4d2]Ed1d3', 'd4Ed1[d3d2]', 'd4EE[d3d2d1]', 'Ed4E[d3d2d1]']
        ]
  
        ## adding the missing transitions in order to have the longest plan in the noisy dataset
        #longest_plan =  ['Ed4E[d3d2d1]', 'Ed4d1[d3d2]', 'E[d4d2]d1d3', 'd3[d4d2]d1E', '[d3d2]d4d1E', '[d3d2]Ed1d4', 'd3d2d1d4', 'Ed2d1[d4d3]', 'EEd1[d4d3d2]', 'EEE[d4d3d2d1]']
        
        
        #

        longest_plans_transis = []

        for longest_plan in longest_plans:

            for i in range(len(longest_plan)-1):
                longest_plans_transis.append(str([longest_plan[i], longest_plan[i+1]]))
            for tr in longest_plans_transis:
                if tr not in all_the_unique_actions_noisy_part:
                    all_the_unique_actions_noisy_part.append(tr)





        all_the_unique_actions_noisy_part_count = {key: 0 for key in all_the_unique_actions_noisy_part}


        # ## REMOVING SOME OF THE ALTERNATIVE PLAN TRANSITIONS from THE NOISY SET
        # longest_plan =  ['Ed4E[d3d2d1]', 'Ed4d1[d3d2]', 'E[d4d2]d1d3', 'd3[d4d2]d1E', '[d3d2]d4d1E', '[d3d2]Ed1d4', 'd3d2d1d4', 'Ed2d1[d4d3]', 'EEd1[d4d3d2]', 'EEE[d4d3d2d1]']
        # #longest_plan_transis = [['Ed4d1[d3d2]', 'E[d4d2]d1d3'], ['d3d2d1d4', 'Ed2d1[d4d3]']]
        # longest_plan_transis = []
        # for i in range(len(longest_plan)-1):
        #     longest_plan_transis.append(str([longest_plan[i], longest_plan[i+1]]))

        # for tr in [longest_plan_transis[1], longest_plan_transis[6]]:
        #     if str(tr) in all_the_unique_actions_noisy_part:
        #         all_the_unique_actions_noisy_part.remove(str(tr))




        if ten_percent_noisy_and_dupplicated:
            #all_the_unique_actions_ten_percent_noisy_and_dup = all_the_unique_actions_noisy_part[:len(all_the_unique_actions_noisy_part)//int(100/18)]
            all_the_unique_actions_ten_percent_noisy_and_dup = all_the_unique_actions_noisy_part[:int(len(all_the_unique_actions_noisy_part) * 1.8 / 100)]
            # 
            # 1% <=> 100
            #  X%   <=> 55
            #   
            
            #  1 <=> 100
            #  X <=> 50    x  =  100/50
            # 100/55



        # check if in all_the_unique_actions_ten_percent_noisy_and_dup there is at least one
        # which belong to longest_plan_transis
        one_belong = False
        for trrrr in all_the_unique_actions_ten_percent_noisy_and_dup:
            if trrrr in longest_plan_transis:
                print("at least one noisy and differentiated action belong to the longest plan")



        all_the_unique_actions_noisy_part_count = {key: 0 for key in all_the_unique_actions_noisy_part}

        print("Number of UNIQUE noisy transs {}".format(str(len(all_the_unique_actions_noisy_part_count)))) # 81


    ###################### COMPUTING THE WHOLE DATASET mean and std #####################
    unique_obs_img = loaded_dataset["unique_obs_img"]
    # array used for computing the whole mean / std all the images (3 copies of each image)
    reduced_uniques = []
    for uniq in unique_obs_img:
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))
    unique_obs_img_preproc, orig_max, orig_min = preprocess(reduced_uniques, histo_bins=24)
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)
    




    longest_plan_transis_test = []

    test_noisy_transis = []

    already_created = False




    ###################### MAIN LOOP #####################

    # loop over traces
    for iii, trace in enumerate(loaded_traces):


        nb_additional_noisy_trans = 0

        actions_for_one_trace = []


        # loop over transitions
        for trtrtr, transitio in enumerate(trace):

            already_created = False

            # normalize the two images
            transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0]), histo_bins=24)
            transi_0_reduced_and_norm, _, _ = normalize_colors(transi_0_prepro, mean=mean_all, std=std_all)

            transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1]), histo_bins=24)
            transi_1_reduced_and_norm, _, _ = normalize_colors(transi_1_prepro, mean=mean_all, std=std_all)
            

            ## stringify both states
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

            #### SAVING THE INIT/GOAL IMAGES (non noisy version)
            if create_init_goal and not already_created:
                already_created = True
                if part1_trans_ == init_st.replace("_noise", ""):
                    print("INIT SAVED")
                    save_np_image(exp_folder, transi_0_reduced_and_norm, "init_NoNoise")
                    # plt.imsave("HANOI_INIT-val.png", reduce_resolution(transitio[0][0]))
                    # plt.close()
                if part2_trans_ == init_st.replace("_noise", ""):
                    print("INIT SAVED")
                    save_np_image(exp_folder, transi_1_reduced_and_norm, "init_NoNoise")
                    # plt.imsave("HANOI_INIT-val.png", reduce_resolution(transitio[0][1]))
                    # plt.close()
                if part1_trans_ == goal_st.replace("_noise", ""):
                    print("GOAL SAVED")
                    save_np_image(exp_folder, transi_0_reduced_and_norm, "goal_NoNoise")
                    # im2 = unnormalize_colors(transi_0_reduced_and_norm, mean_all, std_all)
                    # im2 = deenhance(im2)
                    # im2 = denormalize(im2, orig_min, orig_max)
                    # im2 = np.clip(im2, 0, 1)
                    # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(transitio[0][0]))
                    # plt.close()
                if part2_trans_ == goal_st.replace("_noise", ""):
                    print("GOAL SAVED")
                    save_np_image(exp_folder, transi_1_reduced_and_norm, "goal_NoNoise")



            # If the current transition is not NOISY
            if str([part1_trans_, part2_trans_]) not in all_the_unique_actions_noisy_part_count:

                # add each image to "all_images_reduced_and_norm"
                all_images_reduced_and_norm.append(transi_0_reduced_and_norm)
                all_images_reduced_and_norm.append(transi_1_reduced_and_norm)
                
                # add the transition's label to "actions_for_one_trace"
                actions_for_one_trace.append(str([part1_trans_, part2_trans_]))

                # add the transition's label to "all_actions_unique"
                if str([part1_trans_, part2_trans_]) not in all_actions_unique:

                    all_actions_unique.append(str([part1_trans_, part2_trans_]))
                    all_transitions_unique.append([part1_trans_, part2_trans_])


                # add each image to "all_images_reduced_and_norm_uniques_nonnoisy"
                if str(part1_trans_) not in all_images_reduced_and_norm_uniques_nonnoisy:
                    all_images_reduced_and_norm_uniques_nonnoisy[str(part1_trans_)] = transi_0_reduced_and_norm
                if str(part2_trans_) not in all_images_reduced_and_norm_uniques_nonnoisy:
                    all_images_reduced_and_norm_uniques_nonnoisy[str(part1_trans_)] = transi_1_reduced_and_norm




            # elif the trans is NOISY
            elif add_noisy_trans and str([part1_trans_, part2_trans_]) in all_the_unique_actions_noisy_part_count:

                # update the count (DELETE ??????)
                all_the_unique_actions_noisy_part_count[str([part1_trans_, part2_trans_])] += 1
                

                # create the different noisy versions
                for inndex in range(number_noisy_versions):

                    nb_additional_noisy_trans += 1

                    # create the noised images
                    if 'preprocess' in noise_type:
                        transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0]), histo_bins=24)
                        transi_0_prepro_img_color_norm, _, _ = normalize_colors(transi_0_prepro, mean=mean_all, std=std_all)
                        random.seed(inndex)
                        np.random.seed(inndex)
                        gaussian_noise_0 = np.random.normal(mean, std, transi_0_prepro.shape)
                        random.seed(1)
                        np.random.seed(1)
                        noisy1_reduced_preproc_norm = transi_0_prepro_img_color_norm + gaussian_noise_0
                        noisy1 = noisy1_reduced_preproc_norm

                        transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1]), histo_bins=24)
                        transi_1_prepro_img_color_norm, _, _ = normalize_colors(transi_1_prepro, mean=mean_all, std=std_all)
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


                    # add the two images to "all_images_reduced_and_norm_uniques_noisy" (for generating the PDDL state for exp5)
                    if str(part1_trans_) not in all_images_reduced_and_norm_uniques_noisy:
                        all_images_reduced_and_norm_uniques_noisy[str(part1_trans_)] = noisy1_reduced_preproc_norm

                    if str(part2_trans_) not in all_images_reduced_and_norm_uniques_noisy:
                        all_images_reduced_and_norm_uniques_noisy[str(part1_trans_)] = noisy2_reduced_preproc_norm


                    # if 'noise' init/state was chosen
                    # AND if one of the present states is init/state
                    # save the colored normalized array in some picke file
                    if create_init_goal:

                        if part1_trans_ == init_st.replace("_noise", ""):
                            save_np_image(exp_folder_noisy, noisy1, "init_Noise"+str(inndex))

                        if part2_trans_ == init_st.replace("_noise", ""):
                            save_np_image(exp_folder_noisy, noisy2, "init_Noise"+str(inndex))

                        if part1_trans_ == goal_st.replace("_noise", ""):
                            save_np_image(exp_folder_noisy, noisy1, "goal_Noise"+str(inndex))
                            # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(noisy1))
                            # plt.close()

                        if part2_trans_ == goal_st.replace("_noise", ""):
                            save_np_image(exp_folder_noisy, noisy2, "goal_Noise"+str(inndex))
                            # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(noisy2))
                            # plt.close()


                    # add the noisy images to "all_images_reduced_and_norm" 
                    all_images_reduced_and_norm.append(noisy1_reduced_preproc_norm)
                    all_images_reduced_and_norm.append(noisy2_reduced_preproc_norm)


                    # If the transitions labels must be differentiated !
                    if str([part1_trans_, part2_trans_]) in all_the_unique_actions_ten_percent_noisy_and_dup:

                        # add the label of the transition to "actions_for_one_trace"
                        actions_for_one_trace.append(str([part1_trans_, part2_trans_])+"_"+str(inndex))

                        # add the label of the transition to all_actions_unique and to "all_transitions_unique" 
                        if str([part1_trans_, part2_trans_])+"_"+str(inndex) not in all_actions_unique:

                            all_actions_unique.append(str([part1_trans_, part2_trans_])+"_"+str(inndex))
                            all_transitions_unique.append([part1_trans_+"_"+str(inndex), part2_trans_+"_"+str(inndex)])
                    

                    else:
                        actions_for_one_trace.append(str([part1_trans_, part2_trans_]))

                        # add the action to the unique vectors
                        if str([part1_trans_, part2_trans_]) not in all_actions_unique:

                            all_actions_unique.append(str([part1_trans_, part2_trans_]))
                            all_transitions_unique.append([part1_trans_, part2_trans_])




        # add the actions of this trace to all_actions_for_trace (the array of all actions of all traces)
        all_actions_for_trace.append(actions_for_one_trace)

        # add the indices for this trace to traces_indices array
        traces_indices.append([start_trace_index, start_trace_index+(len(trace)+nb_additional_noisy_trans)*2])
        
        # update the index for the next trace
        start_trace_index+=(len(trace)+nb_additional_noisy_trans)*2







    for k, v in all_images_reduced_and_norm_uniques_noisy.items():
        # print("SHOULD NOT BE HERE")
        # exit()
        save_np_image(exp_folder_all_states, v, "image"+str(k)+"noisy") 



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
    
        for i in range(len(all_actions_unique)):
            #np.argmax(all_actions_one_hot[i]
            f.write("a"+str(i)+" is "+str(all_actions_unique[i])+"\n")


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
        all_pairs_of_images_of_trace_gaussian20, all_pairs_of_images_orig_reduced_of_trace, actions_one_hot_of_trace = modify_one_trace(all_images_transfo_tr_gaussian20, all_images_orig_reduced_tr, all_actions_tr, all_actions_unique)
        all_pairs_of_images_of_trace_gaussian30, _, _ = modify_one_trace(all_images_transfo_tr_gaussian30, all_images_orig_reduced_tr, all_actions_tr, all_actions_unique)
        all_pairs_of_images_of_trace_gaussian40, _, _ = modify_one_trace(all_images_transfo_tr_gaussian40, all_images_orig_reduced_tr, all_actions_tr, all_actions_unique)

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



    print("len all_actions_unique")
    print(len(all_actions_unique))

    print(len(train_set))

    print(train_set[0][0][0].shape)
    print(train_set[0][1].shape)



    return train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min






if args.domain == "hanoi":
    train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = process_dataset(domain="hanoi")
    save_dataset(dataset_folder_name, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min)

elif args.domain == "blocks":
    train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = process_dataset(domain="blocks")
    save_dataset(dataset_folder_name, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min)

elif args.domain == "sokoban":
    train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = process_dataset(domain="sokoban")
    save_dataset(dataset_folder_name, train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min)