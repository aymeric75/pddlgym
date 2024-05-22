import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import json
from concurrent.futures import ProcessPoolExecutor
import random
import torch
import torch.nn.functional as F
from skimage import exposure
import cv2

def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min()), image.max(), image.min()

def equalize(image):
    return exposure.equalize_hist(image, nbins=256)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image):
    image = np.array(image)
    image = image / 255.
    image = image.astype(float)
    image = equalize(image)
    image, orig_max, orig_min = normalize(image)
    image = enhance(image)
    return image, orig_max, orig_min


def gaussian(a, sigma=0.3):
    if sigma == 20:
        np.random.seed(1)
    elif sigma == 30:
        np.random.seed(2)
    elif sigma == 40:
        np.random.seed(3)
    return np.random.normal(0,sigma,a.shape) + a




def deenhance(enhanced_image):
    # Reverse the final shift by subtracting 0.5
    temp_image = enhanced_image - 0.5
    
    # Reverse the clipping: Since clipping limits values, we cannot fully recover original values if they were outside the [-0.5, 0.5] range. 
    # However, for values within the range, we can reverse the scale by dividing by 3.
    # We assume that the enhanced image has values within the range that the clip function allowed.
    temp_image = temp_image / 3
    
    # Reverse the initial shift by adding 0.5 back
    original_image = temp_image + 0.5
    
    return original_image

def denormalize(normalized_image, original_min, original_max):
    if original_max == original_min:
        return normalized_image + original_min
    else:
        return (normalized_image * (original_max - original_min)) + original_min


def unnormalize_colors(normalized_images, mean, std): 
    # Reverse the normalization process
    return (normalized_images*std)+mean



def normalize_colors(images, mean=None, std=None):    
    if mean is None or std is None:
        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)
    return (images - mean)/(std+1e-20), mean, std




# # # Function to reduce resolution of a single image using np.take
# def reduce_resolution(image):
#     # img = cv2.imread("HANOI-tmp.png")
#     res = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_LANCZOS4)
#     return res




# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):
    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 3), axis=0),
                            np.arange(0, image.shape[1], 3), axis=1)

    return reduced_image


def unnormalize_colors(normalized_images, mean, std): 
    # Reverse the normalization process
    unnormalized_images = normalized_images * (std + 1e-6) + mean
    return np.round(unnormalized_images).astype(np.uint8)



# return the origin and destination of a moved disk
def origin_and_destination(action, blocks_data_pre, blocks_data_suc):

    # take the sent disk name
        
    disk_name = action.variables[0].name

    pre_peg = ''
    suc_peg = ''

    # look where it is in the pre lists
    for index, dico in enumerate(blocks_data_pre.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                pre_peg = "peg"+str(index+1)

    # look where it is in the suc list
    for index, dico in enumerate(blocks_data_suc.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                suc_peg = "peg"+str(index+1)

    # return the two "where"
    return pre_peg, suc_peg


# 1401
nb_samplings_per_starting_state = 3001 # has to be ODD 




def generate_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvSokoban55-v0", dynamic_action_space=True)

    all_traces = []
    unique_obs = []
    unique_obs_img = []
    unique_transitions = [] # each ele holds a str representing a unique pair of blocks_data
    obs_occurences = {}
    counter = 0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(0, 31, 1):

        print("ici")

        last_two_blocks_data_str = [] # must contain only two lists that represent a legal transition
        last_two_blocks_data = []
        last_two_imgs = []
        trace_transitions = []

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        # Retrieve the 1st image
        img, blocks_data = env.render()
        img = img[:,:,:3] # remove the transparancy

        # continue
        # print("la")

        plt.imsave("SOKOCRISP.png", img)
        plt.close()

        exit()

        if str(blocks_data) not in unique_obs:
            unique_obs.append(str(blocks_data))
            obs_occurences[str(blocks_data)] = 1
            unique_obs_img.append(img)

        last_two_blocks_data_str.append(str(blocks_data))
        last_two_blocks_data.append(blocks_data)
        last_two_imgs.append(img)

        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_per_starting_state):
            
            if counter%10 == 0:
                print("counter: {}".format(str(counter)))

            # sample an action
            action = env.action_space.sample(obs)

            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)
            img, blocks_data = env.render()
            img = img[:,:,:3]

            last_two_blocks_data_str.append(str(blocks_data))

            # print(blocks_data)
            # print(str(blocks_data))
            # exit()

            last_two_blocks_data.append(blocks_data)
            if len(last_two_blocks_data_str) > 2:
                last_two_blocks_data_str.pop(0)
                last_two_blocks_data.pop(0)

            last_two_imgs.append(img)
            if len(last_two_imgs) > 2:
                last_two_imgs.pop(0)

            if len(last_two_blocks_data_str) == 2:

                if str(last_two_blocks_data_str) not in unique_transitions:

                    # i.e. loose, semi-loose-v1, semi-loose-v2

                    # print("voyons")
                    # #transition_actions.append(str(action))
                    # print(last_two_blocks_data)
                    # print(str(last_two_blocks_data_str))
                    # exit()
        
                    unique_transitions.append(str(last_two_blocks_data_str))
                    trace_transitions.append([[last_two_imgs[0], last_two_imgs[1]], str(last_two_blocks_data_str)])

            if str(blocks_data) not in unique_obs:
                unique_obs.append(str(blocks_data))
                obs_occurences[str(blocks_data)] = 1
                unique_obs_img.append(img)
            else:
                obs_occurences[str(blocks_data)] += 1

            counter += 1

        all_traces.append(trace_transitions)


    print("number of unique transitions is : {}".format(str(len(unique_transitions))))

    with open("resultatSokoban-5-5.txt", 'w') as file2:

        file2.write(str(len(unique_transitions)) + '\n')

    return all_traces, obs_occurences, unique_obs_img, unique_transitions



def generate_dataset_from_list(transitions):

    # all_traces, obs_occurences, unique_obs_img, unique_transitions

    all_traces = []
    obs_occurences = []
    unique_obs_img = []
    unique_transitions = []
    unique_obs = []

    thetrace = []

    env = pddlgym.make("PDDLEnvSokoban55-v0", dynamic_action_space=True)

    obs, debug_info = env.reset(_problem_idx=0)

    for tr in transitions:

        # Retrieve the 1st image
        img1, blocks_data1 = env.render(layout=tr[0], mode='human', close=False)
        img1 = img1[:,:,:3] # remove the transparancy

        # plt.imsave("SOKOCRISP.png", img1)
        # plt.close()
        # exit()

        if str(tr[0]) not in unique_obs:
            unique_obs.append(str(tr[0]))
            unique_obs_img.append(img1)

        # Retrieve the 1st image
        img2, blocks_data2 = env.render(layout=tr[1], mode='human', close=False)
        img2 = img2[:,:,:3] # remove the transparancy
        if str(tr[1]) not in unique_obs:
            unique_obs.append(str(tr[1]))
            unique_obs_img.append(img2)

        thetrace.append([[img1, img2], str(tr[0])+str(tr[1])])

    all_traces.append(thetrace)

    obs_occurences = None
    unique_transitions = None

    return all_traces, obs_occurences, unique_obs_img, unique_transitions


def generate_all_states():

    # 1 c'est l'agent, 2 c'est la caisse

    indices_of_zero = {
        "list1-1": 1,
        "list1-2": 2,
        "list1-3": 3,
        "list2-1": 1,
        "list2-2": 2,
        "list2-3": 3,
        "list3-1": 1,
        "list3-2": 2,
        "list3-3": 3,
    }


    all_states = []



    # loop for the position of the 1
    for kk, in1 in indices_of_zero.items():
        list0 = [5,5,5,5,5]
        list1 = [5,4,0,0,5]
        list2 = [5,0,0,0,5]
        list3 = [5,0,0,0,5]
        list4 = [5,5,5,5,5]


        if "list1" in kk:
            list1[in1] = 1
        elif "list2" in kk:
            list2[in1] = 1
        elif "list3" in kk:
            list3[in1] = 1
        else:
            print("was")
            exit()
        # subloop for the position of the 2
        for kkk, in2 in indices_of_zero.items():
            if kkk != kk:
                if "list1" in kkk:
                    if kkk == "list1-1":
                        list1[in2] = 3
                    else:
                        list1[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4]))
                    if kkk == "list1-1":
                        list1[in2] = 4
                    else:
                        list1[in2] = 0
                elif "list2" in kkk:
                    list2[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4]))
                    list2[in2] = 0
                elif "list3" in kkk:
                    list3[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4]))
                    list3[in2] = 0
                else:
                    print("was2")
                    exit()
            

    return all_states





def generate_next_states(state):

    
    # state = np.array([
    #     [5,5,5,5,5],
    #     [5,4,0,2,5],
    #     [5,0,1,0,5],
    #     [5,0,0,0,5],
    #     [5,5,5,5,5]
    # ])

    # where is 1 ? where is 2 ?
    index1 = [np.where(state == 1)[1][0], np.where(state == 1)[0][0]]

    index2 = None
    if len(np.where(state == 2)[0]) > 0:
        index2 = [np.where(state == 2)[1][0], np.where(state == 2)[0][0]]


    # depending on where is 1, what are the next possibles positions ?
    next_moves_for_1 = {

        '[1, 1]' : [[2, 1], [1, 2]],
        '[2, 1]' : [[1, 1], [2, 2], [3, 1]],
        '[3, 1]' : [[2, 1], [3, 2]],

        '[1, 2]' : [[1, 1], [2, 2], [1, 3]],
        '[2, 2]' : [[2, 1], [3, 2], [2, 3], [1, 2]],
        '[3, 2]' : [[3, 1], [2, 2], [3, 3]],

        '[1, 3]' : [[1, 2], [2, 3]],
        '[2, 3]' : [[1, 3], [2, 2], [3, 3]],
        '[3, 3]' : [[3, 2], [2, 3]],
        
    }



    # if 2 on one of the next possible position ? can it be moved ? yes/no, if no, pos cannot change
    # if yes, find where it can be moved
    # print("la")
    # print(next_moves_for_1[str([index1[0], index1[1]])])
    
    next_states = []

    # for all "possible" next states for "1"
    for next_1 in next_moves_for_1[str([index1[0], index1[1]])]:

        if index2 is not None:

            # if 2 on the next possible move
            if str(next_1) == str([index2[0], index2[1]]):

                #print("current pos is {}".format(str([index1[0], index1[1]])))

                #print("next pos is {}".format(str(next_1)))


                diff = np.array(next_1) - np.array([index1[0], index1[1]])

                move_numb = diff[diff != 0][0]
                # if move < 0 = go right or down
                # if move > 0 = go left or up
                index_diff_zero = np.where(diff != 0)[0][0]


                move = ""
                if move_numb < 0 and index_diff_zero == 0:
                    move = 'left'
                    possible_next_2 = [index2[0]-1, index2[1]]
                elif move_numb < 0 and index_diff_zero == 1:
                    move = 'up'
                    possible_next_2 = [index2[0], index2[1]-1]
                elif move_numb > 0 and index_diff_zero == 0:
                    move = 'right'
                    possible_next_2 = [index2[0]+1, index2[1]]
                elif move_numb > 0 and index_diff_zero == 1:
                    move = 'down'
                    possible_next_2 = [index2[0], index2[1]+1]

                #print("move : {}".format(move))
            
                # compute the next spot of 2, if outside the range MEANS move NOT possible
                # 
                # 

                if 0 in possible_next_2 or 4 in possible_next_2:
                    # then move not possible
                    # DONT ADD THE STATE 
                    continue

                else:

                    empty_state = np.array([
                        [5,5,5,5,5],
                        [5,4,0,0,5],
                        [5,0,0,0,5],
                        [5,0,0,0,5],
                        [5,5,5,5,5]
                    ])

                    # new position 1
                    empty_state[next_1[1], next_1[0]] = 1

                    #  new position 2
                    #       if 1,1 then turn 1,1 into 3
                    if possible_next_2[0] == 1 and possible_next_2[1] == 1:
                        empty_state[1, 1] = 3
                    else:
                        empty_state[possible_next_2[1], possible_next_2[0]] = 2
                    #   
                    
                    next_states.append(empty_state)

            else:

                next_state = state.copy()

                next_state[next_1[1], next_1[0]] = 1
                if index1[1] == 1 and index1[0] == 1:
                    next_state[index1[1], index1[0]] = 4
                else:
                    next_state[index1[1], index1[0]] = 0

                # 
                next_states.append(next_state)

        else:

            next_state = state.copy()

            next_state[next_1[1], next_1[0]] = 1
            
            if index1[1] == 1 and index1[0] == 1:
                next_state[index1[1], index1[0]] = 4
            else:
                next_state[index1[1], index1[0]] = 0

            # 
            next_states.append(next_state)

    #print(next_states)    
    return next_states



def generate_all_transitions():

    all_transitions = []

    for state in generate_all_states():

        next_states = generate_next_states(state)

        for next_ in next_states:
            
            all_transitions.append([state, next_])


    return all_transitions




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
            all_pairs_of_images_orig.append([all_images_orig_tr[iiii], all_images_orig_tr[iiii+1]])

    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions:
        all_actions_indices.append(all_actions_unique_.index(str(ac)))
    
    actions_indexes = torch.tensor(all_actions_indices).long()

    # array of one hot encoded actions
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique_))

    return all_pairs_of_images, all_pairs_of_images_orig, actions_one_hot.numpy()


def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def save_dataset(dire, traces, obs_occurences, unique_obs_img, unique_transitions):
    data = {
        "traces": traces,
        "obs_occurences": obs_occurences,
        "unique_obs_img": unique_obs_img,
        "unique_transitions": unique_transitions
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = "data.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)


# # 1) generate dataset (only once normally)
# all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_dataset()


# # # 2) save dataset
# # save_dataset("sokoban_5-5_dataset", all_traces, obs_occurences, unique_obs_img, unique_transitions)

# exit()


def create_a_trace():
    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/data.p")

    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):
        for trtrtr, transitio in enumerate(trace):
            plt.imsave("hanoi_pair_"+str(trtrtr)+"_pre.png", reduce_resolution(transitio[0][0]))
            plt.close()
            plt.imsave("hanoi_pair_"+str(trtrtr)+"_suc.png", reduce_resolution(transitio[0][1]))
            plt.close()
            if trtrtr > 7:
                break
        break
    return



def stringify_state(state):

    str_version = '['
    for i, line in enumerate(state):
        if i < 4:
            str_version += str(line)+' '
        else:
            str_version += str(line)
    str_version += ']'
    return str_version

def stringify_transition(state1, state2):

    final = "['"

    final += stringify_state(state1)

    final += "', '"

    final += stringify_state(state2)

    final += "']"

    return final


# # # # 1)
# all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_dataset_from_list(generate_all_transitions())



# # 2) save dataset
# save_dataset("sokoban_5-5_dataset", all_traces, obs_occurences, unique_obs_img, unique_transitions)


# exit()


import re

# # Regular expression to find the pattern [LETTERNUMBER]
# pattern = r"\[([a-zA-Z])([0-9])\]"

# # Function to replace the pattern [LETTERNUMBER] with LETTERNUMBER
# def replace_pattern(match):
#     return f"{match.group(1)}{match.group(2)}"




def save_np_image(dire, array, file_name):
    data = {
        "image": array,
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = str(file_name)+".p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)

def replace_in_str(word):
    
    res1 = word.replace("5", "")
    res1 = res1.replace("[", "")
    res1 = res1.replace("]", "")
    res1 = res1.replace(" ", "")
    res1 = res1.replace("\n", "")
  
    return [res1[:9],res1[9:]]


def export_dataset(action_type="full"):

    orig_min = None
    orig_max = None

    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban_5-5_dataset/data.p")

    #all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_dataset_from_list(generate_all_transitions())
    alltr = loaded_dataset["traces"]

    print(len(alltr))

    # for iiiii, trr in enumerate(alltr[0]):

    #     print(type(trr[0]))
    #     print(np.array(trr[0]).shape)

    #     #combined_image = np.concatenate((reduce_resolution(trr[0][0]), reduce_resolution(trr[0][1])), axis=1)
    #     plt.imsave("Soko-transi"+str(iiiii)+"_0.png", reduce_resolution(trr[0][0]))
    #     plt.imsave("Soko-transi"+str(iiiii)+"_1.png", reduce_resolution(trr[0][1]))

    # exit()


    all_images_reduced = []
    all_actions = []
    all_pairs_of_images = []
    all_pairs_of_images_reduced_orig = []
    all_actions_one_hot = []
    all_actions_unique = []

    traces_indices = []
    traces_actions_indices = []
    start_trace_index = 0
    start_trace_action_index = 0
    all_actions_for_trace = []
    unique_actions = []

    all_transitions_unique = []

    total_number_transitions = 0

    # first loop to compute the whole dataset mean and std
    #for iii, trace in enumerate(all_traces):
    for iii, trace in enumerate(loaded_dataset["traces"]):
    
        traces_indices.append([start_trace_index, start_trace_index+len(trace)*2])
        start_trace_index+=len(trace)*2

        traces_actions_indices.append([start_trace_action_index, start_trace_action_index+len(trace)])
        start_trace_action_index+=len(trace)
        
        actions_for_one_trace = []

        for trtrtr, transitio in enumerate(trace):

            total_number_transitions += 1

            all_images_reduced.append(reduce_resolution(transitio[0][0])) # = im1
            all_images_reduced.append(reduce_resolution(transitio[0][1])) # = im2

            if trtrtr < 10:

                plt.imsave("hanoii_tr_"+str(iii)+"_transi_"+str(trtrtr)+"_0.png", reduce_resolution(transitio[0][0]))
                plt.close()

                plt.imsave("hanoii_tr_"+str(iii)+"_transi_"+str(trtrtr)+"_1.png", reduce_resolution(transitio[0][1]))
                plt.close()
            
            # print("loooo")
            # print(transitio[1])
            # print("laaaaaa")
            # print(replace_in_str(transitio[1]))
            # exit()

            if transitio[1] not in unique_actions:

                #combined_image = np.concatenate((reduce_resolution(transitio[0][0]), reduce_resolution(transitio[0][1])), axis=1)
                #plt.imsave("Soko-transi"+str(len(unique_actions))+".png", combined_image)
                #print("action for transi "+str(len(unique_actions))+" is "+transitio[1])


                # combined_image = np.concatenate((reduce_resolution(transitio[0][0]), reduce_resolution(transitio[0][1])), axis=1)
                # plt.imsave("all_sokos/soko-"+str(replace_in_str(transitio[1]))+".png", combined_image)
                # print("action for transi "+str(len(unique_actions))+" is "+transitio[1])

                unique_actions.append(transitio[1])

                traaaa = replace_in_str(transitio[1])


                all_transitions_unique.append([[transitio[0][0], transitio[0][1]], [traaaa[0], traaaa[1]] ])


            all_actions.append(transitio[1])
            actions_for_one_trace.append(transitio[1])

        all_actions_for_trace.append(actions_for_one_trace)














































    unique_obs_img = loaded_dataset["unique_obs_img"]
    


    print("total_number_transitions : {}".format(str(total_number_transitions)))
    

    print("size unique_actions: {}".format(str(len(unique_actions))))


    reduced_uniques = []
    for uniq in unique_obs_img:

        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 20), 0, 255))
        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 30), 0, 255))
        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 40), 0, 255))
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))

        plt.imsave("ssssssSSSSsoko.png", reduce_resolution(uniq))
        print(reduce_resolution(uniq).shape)
        #exit()


    #unique_obs_img_preproc, orig_max, orig_min = preprocess(reduced_uniques)
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(reduced_uniques, mean=None, std=None)


















    ############################################
    ######## Building the file for the DFA #####
    ############################################

    # Input:
    # all_actions_unique
    # 
    print(len(all_transitions_unique))


    # WE NEED
    # initial_state
    # final_state
    # alphabet_dic (for each transition (key) we associate a name ("a"+str(i)))
    # unique_states
    # all_transitions where each item is the transi described as firstState+nameAction+SecondState

    total_dfa_dico = {}
    initial_state = all_transitions_unique[0][1][0]
    final_state = all_transitions_unique[-1][1][1]
    print(type(initial_state))
    print(final_state)

    alphabet_dic = {}
    unique_states = []
    all_transitions_dfa = []

    # print the init/goal from here
    # 
    # 
    #init_st = "[d4d3d2d1]EEE"
    # , 
    init_st = '120000000'
    goal_st = '300000100'


    # 

    for iu, tran in enumerate(all_transitions_unique):
        
        alphabet_dic[str(tran[1])] = "a"+str(iu)
        print("laza")
        print(tran[1][0])
        if tran[1][0] not in unique_states:
            unique_states.append(str(tran[1][0]))

        if tran[1][1] not in unique_states:
            unique_states.append(str(tran[1][1]))

        normi = normalize_colors(reduce_resolution(tran[0][0]), mean=mean_all, std=std_all)

        if tran[1][0] == init_st:
            
            save_np_image("sokoban_5-5_dataset", normi, "init_1")
            plt.imsave("SOKO_INIT.png", reduce_resolution(tran[0][0]))
            plt.close()

        if tran[1][1] == init_st:
            save_np_image("sokoban_5-5_dataset", reduce_resolution(tran[0][0]), "init_1")
            plt.imsave("SOKO_INIT.png", reduce_resolution(tran[0][1]))
            plt.close()

        if tran[1][0] == goal_st:
            save_np_image("sokoban_5-5_dataset", reduce_resolution(tran[0][0]), "goal_1")
            plt.imsave("SOKO_GOAL.png", reduce_resolution(tran[0][0]))
            plt.close()

        if tran[1][1] == goal_st:
            save_np_image("sokoban_5-5_dataset", reduce_resolution(tran[0][0]), "goal_1")
            plt.imsave("SOKO_GOAL.png", reduce_resolution(tran[0][1]))
            plt.close()


    for iu, tran in enumerate(all_transitions_unique):
        transi = []
        transi.append(str(tran[1][0]))
        transi.append(alphabet_dic[str(tran[1])])
        transi.append(str(tran[1][1]))
        all_transitions_dfa.append(transi)


    total_dfa_dico["alphabet"] = list(alphabet_dic.values())
    total_dfa_dico["states"] = unique_states
    total_dfa_dico["initial_state"] = initial_state
    #total_dfa_dico["initial_state"] = ""
    total_dfa_dico["accepting_states"] = final_state
    #total_dfa_dico["accepting_states"] = "" # final_state
    total_dfa_dico["transitions"] = all_transitions_dfa

    with open("DFA_Soko_5_5.json", 'w') as f:
        json.dump(total_dfa_dico, f)

    ############################################
    #### END Building the file for the DFA #####
    ############################################












    # all_images_reduced > gaussian > clip > preprocess > normalize_colors


    # plt.imsave("SUKI.png", all_images_reduced[0])
    # plt.close()
    # exit()


    all_images_reduced = np.array(all_images_reduced)
    # # 
    # all_images_reduced_gaussian_20 = np.clip(gaussian(all_images_reduced, 20), 0, 255)
    # all_images_reduced_gaussian_30 = np.clip(gaussian(all_images_reduced, 30), 0, 255)
    # all_images_reduced_gaussian_40 = np.clip(gaussian(all_images_reduced, 40), 0, 255)

    #all_images_reduced_gaussian_20_preproc, _, _ = preprocess(all_images_reduced)
    all_images_reduced_gaussian_20_norm, __, __ = normalize_colors(all_images_reduced, mean=mean_all, std=std_all)
    

    #all_images_reduced_gaussian_30_preproc, _, _ = preprocess(all_images_reduced)
    all_images_reduced_gaussian_30_norm, __, __ = normalize_colors(all_images_reduced, mean=mean_all, std=std_all)

    #all_images_reduced_gaussian_40_preproc, _, _ = preprocess(all_images_reduced)
    all_images_reduced_gaussian_40_norm, __, __ = normalize_colors(all_images_reduced, mean=mean_all, std=std_all)



    print("IMAGE DATAS")

    print(np.max(np.unique(all_images_reduced_gaussian_20_norm)))
    print(np.max(all_images_reduced_gaussian_20_norm))
    print(np.min(all_images_reduced_gaussian_20_norm))



    #build array containing all actions WITHOUT DUPLICATE
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))


    all_pairs_of_images_processed_gaussian20 = []
    all_pairs_of_images_processed_gaussian30 = []
    all_pairs_of_images_processed_gaussian40 = []


    # second loop to process the pairs
    for iiii, trace in enumerate(loaded_dataset["traces"]):


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
    


    return train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min


#train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()

# exit()

# print(generate_all_states()[:10])


# print("len train_set")

# print(len(train_set))


# for hh in range(0, len(train_set), 10):

#     acc = all_actions_unique[np.argmax(train_set[hh][1])]
#     print("action for {} is {}".format(str(hh), str(acc)))

#     # im1_orig=all_pairs_of_images_reduced_orig[hh][0]
#     # im2_orig=all_pairs_of_images_reduced_orig[hh][1]
#     im1 = train_set[hh][0][0]
#     im2 = train_set[hh][0][1]

#     im1 = unnormalize_colors(im1, mean_all, std_all)
#     im1 = deenhance(im1)
#     im1 = denormalize(im1, orig_min, orig_max)
#     im1 = np.clip(im1, 0, 1)
    
#     im2 = unnormalize_colors(im2, mean_all, std_all)
#     im2 = deenhance(im2)
#     im2 = denormalize(im2, orig_min, orig_max)
#     im2 = np.clip(im2, 0, 1)


#     plt.imsave("blocks_3_pair_"+str(hh)+"_pre.png", im1)
#     plt.close()

#     plt.imsave("blocks_3_pair_"+str(hh)+"_suc.png", im2)
#     plt.close()


# exit()


