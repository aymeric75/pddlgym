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

def preprocess(image, histo_bins=24):
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


# def unnormalize_colors(normalized_images, mean, std): 
#     # Reverse the normalization process
#     unnormalized_images = normalized_images * (std + 1e-6) + mean
#     return np.round(unnormalized_images).astype(np.uint8)



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

        # plt.imsave("SOKOCRISP.png", img)
        # plt.close()

        # exit()    Sokoban_6_6_FullDFA300000000001000000000000000000000000-300001000000000000000000000000000000

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
    #print(obs)
    #exit()
    #img, blocks_data = env.render()



    # exit()

    for innn, tr in enumerate(transitions):


        # print("tr[0]tr[0]tr[0]tr[0]")
        # print(tr[0])
        # print('ttttttttt11111')


        # Retrieve the 1st image
        img1, blocks_data1 = env.render(layout=tr[0], mode='human', close=False)
        img1 = img1[:,:,:3] # remove the transparancy

        # if innn > 2:
        #     plt.imsave("LE_SOKO2.png", img1)
        #     plt.close()
        #     exit()


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
    # 
    indices_of_zero = {

        "list1-1": 1,
        "list1-2": 2,
        "list1-3": 3,
        "list1-4": 4,
        "list1-5": 5,
        "list1-6": 6,
        
        "list2-1": 1,
        "list2-2": 2,
        "list2-3": 3,
        "list2-4": 4,
        "list2-5": 5,
        "list2-6": 6,

        "list3-1": 1,
        "list3-2": 2,
        "list3-3": 3,
        "list3-4": 4,
        "list3-5": 5,
        "list3-6": 6,

        "list4-1": 1,
        "list4-2": 2,
        "list4-3": 3,
        "list4-4": 4,
        "list4-5": 5,
        "list4-6": 6,

        "list5-1": 1,
        "list5-2": 2,
        "list5-3": 3,
        "list5-4": 4,
        "list5-5": 5,
        "list5-6": 6,


        "list6-1": 1,
        "list6-2": 2,
        "list6-3": 3,
        "list6-4": 4,
        "list6-5": 5,
        "list6-6": 6,

    }


    all_states = []



    # loop for the position of the 1
    for kk, in1 in indices_of_zero.items():

        list0 = [5,5,5,5,5,5,5,5]
        list1 = [5,4,0,0,0,0,0,5]
        list2 = [5,0,0,0,0,0,0,5]
        list3 = [5,0,0,0,0,0,0,5]
        list4 = [5,0,0,0,0,0,0,5]
        list5 = [5,0,0,0,0,0,0,5]
        list6 = [5,0,0,0,0,0,0,5]
        list7 = [5,5,5,5,5,5,5,5]


        if "list1" in kk:
            list1[in1] = 1
        elif "list2" in kk:
            list2[in1] = 1
        elif "list3" in kk:
            list3[in1] = 1
        elif "list4" in kk:
            list4[in1] = 1
        elif "list5" in kk:
            list5[in1] = 1
        elif "list6" in kk:
            list6[in1] = 1
        else:
            print("was")
            exit()

        # 1: agent
        # 2: caisse
        # 3: caisse on  goal
        # 4: goal empty

        # "kk" : pos of the agent
        # "kkk" : pos of the caisse

        # subloop for the position of the 2
        for kkk, in2 in indices_of_zero.items():
            if kkk != kk:
                if "list1" in kkk: # 
                    if kkk == "list1-1": # if la caisse est en haut à gauche
                        list1[in2] = 3 # on place le "3" en haut à gauche
                    else: # on place la caisse à son indice (en haut, mais pas à gauche), donc on met un "2"
                        list1[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4,list5,list6,list7]))
                    if kkk == "list1-1": # une fois le state ajouté, on remet la ligne "1" dans son état originel (avec un 4 à gauche et des 0)
                        list1[in2] = 4
                    else:
                        list1[in2] = 0
                elif "list2" in kkk:
                    list2[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4,list5,list6,list7]))
                    list2[in2] = 0
                elif "list3" in kkk:
                    list3[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4,list5,list6,list7]))
                    list3[in2] = 0
                elif "list4" in kkk:
                    list4[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4,list5,list6,list7]))
                    list4[in2] = 0
                elif "list5" in kkk:
                    list5[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4,list5,list6,list7]))
                    list5[in2] = 0
                elif "list6" in kkk:
                    list6[in2] = 2
                    all_states.append(np.array([list0,list1,list2,list3,list4,list5,list6,list7]))
                    list6[in2] = 0
                else:
                    print("was2")
                    exit()
            

    return all_states




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



def replace_in_str(word):
    
    res1 = word.replace("5", "")
    res1 = res1.replace("[", "")
    res1 = res1.replace("]", "")
    res1 = res1.replace(" ", "")
    res1 = res1.replace("\n", "")
    

    return [res1[:len(res1)//2], res1[len(res1)//2:]]


# when word containes only one state (not a transition)
def replace_in_str_simple(word):
    word = str(word)
    res1 = word.replace("5", "")
    res1 = res1.replace("[", "")
    res1 = res1.replace("]", "")
    res1 = res1.replace(" ", "")
    res1 = res1.replace("\n", "")
    

    return res1






def generate_next_states(state):

    
    # state = np.array([
    #     [5,5,5,5,5,5,5,5],
    #     [5,4,0,2,0,0,0,5],
    #     [5,0,1,0,0,0,0,5],
    #     [5,0,0,0,0,0,0,5],
    #     [5,0,0,0,0,0,0,5],
    #     [5,0,0,0,0,0,0,5],
    #     [5,0,0,0,0,0,0,5],
    #     [5,5,5,5,5,5,5,5]
    # ])

    # where is 1 ? where is 2 ?
    # 1 is the agent
    index1 = [np.where(state == 1)[1][0], np.where(state == 1)[0][0]]

    index2 = None
    if len(np.where(state == 2)[0]) > 0:
        index2 = [np.where(state == 2)[1][0], np.where(state == 2)[0][0]]


    # depending on where is 1, what are the next possibles positions ?
    # [x, y]
    # go 
    next_moves_for_1 = {

        # 1st row    BON
        '[1, 1]' : [[2, 1], [1, 2]],
        '[2, 1]' : [[1, 1], [2, 2], [3, 1]],
        '[3, 1]' : [[2, 1], [4, 1], [3, 2]],
        '[4, 1]' : [[3, 1], [4, 2], [5, 1]],
        '[5, 1]' : [[4, 1], [5, 2], [6, 1]],
        '[6, 1]' : [[5, 1], [6, 2]],

        # 2nd row       BON 
        '[1, 2]' : [[1, 1], [2, 2], [1, 3]],
        '[2, 2]' : [[1, 2], [2, 1], [3, 2], [2, 3]],
        '[3, 2]' : [[2, 2], [3, 1], [4, 2], [3, 3]],
        '[4, 2]' : [[3, 2], [4, 1], [5, 2], [4, 3]],
        '[5, 2]' : [[4, 2], [5, 1], [6, 2], [5, 3]],
        '[6, 2]' : [[5, 2], [6, 1], [6, 3]],

        # 3rd row          BON
        '[1, 3]' : [[1, 2], [2, 3], [1, 4]],
        '[2, 3]' : [[1, 3], [2, 2], [3, 3], [2, 4]],
        '[3, 3]' : [[2, 3], [3, 2], [4, 3], [3, 4]],
        '[4, 3]' : [[3, 3], [4, 2], [5, 3], [4, 4]],
        '[5, 3]' : [[4, 3], [5, 2], [6, 3], [5, 4]],
        '[6, 3]' : [[5, 3], [6, 2], [6, 4]],
    
        # 4th row       BON     
        '[1, 4]' : [[1, 3], [2, 4], [1, 5]],
        '[2, 4]' : [[1, 4], [2, 3], [3, 4], [2, 5]],
        '[3, 4]' : [[2, 4], [3, 3], [4, 4], [3, 5]],
        '[4, 4]' : [[3, 4], [4, 3], [5, 4], [4, 5]],
        '[5, 4]' : [[4, 4], [5, 3], [6, 4], [5, 5]],
        '[6, 4]' : [[5, 4], [6, 3], [6, 5]],

        # 5th row           BON
        '[1, 5]' : [[1, 4], [2, 5], [1, 6]],
        '[2, 5]' : [[1, 5], [2, 4], [3, 5], [2, 6]],
        '[3, 5]' : [[2, 5], [3, 4], [4, 5], [3, 6]],
        '[4, 5]' : [[3, 5], [4, 4], [5, 5], [4, 6]],
        '[5, 5]' : [[4, 5], [5, 4], [6, 5], [5, 6]],
        '[6, 5]' : [[5, 5], [6, 4], [6, 6]],
    
        # 6th row           BON
        '[1, 6]' : [[1, 5], [2, 6]],
        '[2, 6]' : [[1, 6], [2, 5], [3, 6]],
        '[3, 6]' : [[2, 6], [3, 5], [4, 6]],
        '[4, 6]' : [[3, 6], [4, 5], [5, 6]],
        '[5, 6]' : [[4, 6], [5, 5], [6, 6]],
        '[6, 6]' : [[5, 6], [6, 5]],


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
                        [5,5,5,5,5,5,5,5],
                        [5,4,0,0,0,0,0,5],
                        [5,0,0,0,0,0,0,5],
                        [5,0,0,0,0,0,0,5],
                        [5,0,0,0,0,0,0,5],
                        [5,0,0,0,0,0,0,5],
                        [5,0,0,0,0,0,0,5],
                        [5,5,5,5,5,5,5,5]
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

def find_index_2d(array, target):
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            if value == target:
                return (i, j)
    return None  # 

def generate_all_transitions():

    all_transitions = []


    for state in generate_all_states():

        next_states = generate_next_states(state)


        for next_ in next_states:


            if str(find_index_2d(next_, 2)) == str(find_index_2d(state, 1)):
                continue


            all_transitions.append([state, next_])
            # print("transiti")
            # print([state, next_])

            # DONT ADD THE TRANSITION IF dans _NEXT posOf2 is equal to posOf1 dans state

            # print("STATE")
            # print(state)
            # print("NEXT")
            # print(type(next_))



            #exit()

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
            #all_pairs_of_images_orig.append([all_images_orig_tr[iiii], all_images_orig_tr[iiii+1]])

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



# # 2) save dataset
# save_dataset("sokoban_6-6_dataset", all_traces, obs_occurences, unique_obs_img, unique_transitions)

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





# # 1)
# all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_dataset_from_list(generate_all_transitions())

# # 2) save dataset
# save_dataset("sokoban_6-6_dataset_NoIllegal", all_traces, obs_occurences, unique_obs_img, unique_transitions)



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



def export_dataset(action_type="full",
        remove_a_trans=False,
        remove_some_trans=False,
        build_dfa=False,
        add_noisy_trans=True,
        ten_percent_noisy_and_dupplicated=True,
        noise_type='preprocess',
        number_noisy_versions=3,
        init_st = "402000000000000100000000000000000000",
        goal_st = "402000000000000010000000000000000000",
        create_init_goal=True,
        perc=55,
        std=0.015
        ):


    exp_folder = None
    exp_folder_ten_percent_noisy_and_dupplicated = None

    if not os.path.exists("Sokoban_6_6_FullDFA"+init_st+"-"+goal_st):
        os.makedirs("Sokoban_6_6_FullDFA"+init_st+"-"+goal_st) 
    exp_folder = "Sokoban_6_6_FullDFA"+init_st+"-"+goal_st
    global counntteerr

    trans_to_remove_list = [init_st, goal_st]
    all_the_unique_actions_ten_percent_noisy_and_dup = []
    all_the_unique_actions_noisy_part_count = {}
    transis_to_remove = []


    if remove_some_trans:
        loaded_removable_trans = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban6_6_missing_edges_3/deleted_edges.p")
        transis_to_remove = loaded_removable_trans["stuff"]

        loaded_removable_edge_goal = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban6_6_missing_edges_3/goal_state.p")
        edge_removed_goal = loaded_removable_edge_goal["stuff"]
        goal_st = edge_removed_goal

        loaded_removable_edge_init = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban6_6_missing_edges_3/init_state.p")
        edge_removed_init = loaded_removable_edge_init["stuff"]
        init_st = edge_removed_init

    
    transis_to_remove_str = []
    for tr in transis_to_remove:
        transis_to_remove_str.append(str(tr))


    if not os.path.exists("sokoban-6-6_FullDFA-no-noise-"+init_st+"-"+goal_st):
        os.makedirs("sokoban-6-6_FullDFA-no-noise-"+init_st+"-"+goal_st) 
    exp_folder_ = "sokoban-6-6_FullDFA-no-noise-"+init_st+"-"+goal_st



    if ten_percent_noisy_and_dupplicated:
        if not os.path.exists("ten_percent_noisy_and_dupplicated"):
            os.makedirs("ten_percent_noisy_and_dupplicated") 
        exp_folder_ten_percent_noisy_and_dupplicated = "ten_percent_noisy_and_dupplicated"      

    if add_noisy_trans:
        # creating init/goal dir for the experiment
        if not os.path.exists("sokoban_6_6_perc"+str(perc)+"_#v"+str(number_noisy_versions)+"_std"+str(std)):
            os.makedirs("sokoban_6_6_perc"+str(perc)+"_#v"+str(number_noisy_versions)+"_std"+str(std)) 
        exp_folder = "sokoban_6_6_perc"+str(perc)+"_#v"+str(number_noisy_versions)+"_std"+str(std)

    elif remove_a_trans:
        # creating init/goal dir for the experiment
        if not os.path.exists("sokoban_6_6_RemovedTrans-"+trans_to_remove_list[0]+"-"+trans_to_remove_list[1]):
            os.makedirs("sokoban_6_6_RemovedTrans-"+trans_to_remove_list[0]+"-"+trans_to_remove_list[1]) 
        exp_folder = "sokoban_6_6_RemovedTrans-"+trans_to_remove_list[0]+"-"+trans_to_remove_list[1]

    elif remove_some_trans:
        if not os.path.exists("sokoban_6_6_missing_edges_3"):
            os.makedirs("sokoban_6_6_missing_edges_3") 
        exp_folder = "sokoban_6_6_missing_edges_3"


    # mean and std for the noise
    mean = 0

    #loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban_6-6_dataset/data.p")
    all_images_reduced_and_norm_uniques_nonnoisy = {}
    all_images_reduced_and_norm_uniques_noisy = {}
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban_6-6_dataset_NoIllegal/data.p")

    all_images_reduced = [] # all the raw images in the order listed by the traces
    all_images_reduced_and_norm = []
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


    loaded_traces = loaded_dataset["traces"]

    print("BEFORE")

    print(len(transis_to_remove_str)) # 519 


    #
    for iii, trace in enumerate(loaded_traces):
        for trtrtr, transitio in enumerate(trace):
            if str(replace_in_str(transitio[1])) not in all_the_unique_actions:
                traaaa = replace_in_str(transitio[1])
                all_the_unique_actions.append(str(traaaa))

    # print(len(all_the_unique_actions))
    # exit()

    ## 
    if remove_some_trans:
        loaded_traces_after_prunning = []
        for iii, trace in enumerate(loaded_traces):
            #trace_copy = trace.copy()
            if len(trace) == 0:
               continue
            trace_tmp = []
            for trtrtr, transitio in enumerate(trace):
                # print("replace_in_str(transitio[1])")
                # print(replace_in_str(transitio[1]))
                # exit()
                if str(replace_in_str(transitio[1])) not in transis_to_remove_str:
                    if str(replace_in_str(transitio[1])) != str(trans_to_remove_list):
                        # print(str(replace_in_str(transitio[1])))
                        # print()
                        # print(str(trans_to_remove_list))
                        # exit()
                        trace_tmp.append(transitio)

            loaded_traces_after_prunning.append(trace_tmp)
        loaded_traces = loaded_traces_after_prunning



    if add_noisy_trans:

        random.shuffle(all_the_unique_actions)

        ## taking X % of the transitions (that are then noised)
        all_the_unique_actions_noisy_part = all_the_unique_actions[:int(len(all_the_unique_actions) * perc / 100)]



        longest_plans = [
            ['402000000000000000000000000000100000', '402000000000000000000000100000000000', '402000000000000000100000000000000000', '402000000000100000000000000000000000', '402000000000010000000000000000000000', '402000001000000000000000000000000000', '402000000100000000000000000000000000', '402100000000000000000000000000000000', '421000000000000000000000000000000000', '310000000000000000000000000000000000', '301000000000000000000000000000000000', '300000001000000000000000000000000000', '300000000000001000000000000000000000', '300000000000000100000000000000000000', '300000000000000000000100000000000000', '300000000000000000000000000100000000', '300000000000000000000000000000000100', '300000000000000000000000000000000010', '300000000000000000000000000000000001'],
            ['420000000000000000000000000001000000', '420000000000000000000001000000000000', '420000000000000001000000000000000000', '420000000000000010000000000000000000', '420000000010000000000000000000000000', '420000000100000000000000000000000000', '420000001000000000000000000000000000', '420000010000000000000000000000000000', '420000100000000000000000000000000000', '120000000000000000000000000000000000', '412000000000000000000000000000000000', '402000010000000000000000000000000000', '402000001000000000000000000000000000', '402000000000001000000000000000000000', '402000000000000100000000000000000000', '402000000000000000000100000000000000', '402000000000000000000010000000000000', '402000000000000000000000000010000000', '402000000000000000000000000000000010', '402000000000000000000000000000000001'],
            ['420000000000000000000000000000000001', '420000000000000000000000000000000010', '420000000000000000000000000000000100', '420000000000000000000000000000001000', '420000000000000000000000001000000000', '420000000000000000001000000000000000', '420000000000000000010000000000000000', '420000000000010000000000000000000000', '420000000000100000000000000000000000', '420000100000000000000000000000000000', '120000000000000000000000000000000000', '412000000000000000000000000000000000', '402000010000000000000000000000000000', '402000001000000000000000000000000000', '402000000000001000000000000000000000', '402000000000000100000000000000000000', '402000000000000000000100000000000000', '402000000000000000000010000000000000', '402000000000000000000000000010000000', '402000000000000000000000000000000010', '402000000000000000000000000000000001']
        ]
  



        longest_plans_transis = []

        for longest_plan in longest_plans:

            for i in range(len(longest_plan)-1):
                longest_plans_transis.append(str([longest_plan[i], longest_plan[i+1]]))
            for tr in longest_plans_transis:
                if tr not in all_the_unique_actions_noisy_part:
                    all_the_unique_actions_noisy_part.append(tr)




        all_the_unique_actions_noisy_part_count = {key: 0 for key in all_the_unique_actions_noisy_part}

        print("Number of UNIQUE noisy trans {}".format(str(len(all_the_unique_actions_noisy_part_count)))) # 81

        # print("all_the_unique_actions")
        # print(len(all_the_unique_actions))
        # print(len(all_the_unique_actions_noisy_part))
        # exit()



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
    



    # COMPUTING THE WHOLE DATASET mean and std
    unique_obs_img = loaded_dataset["unique_obs_img"]

    # array used for computing the whole mean / std all the images (3 copies of each image)
    reduced_uniques = []
    for uniq in unique_obs_img:
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))


    # for uniq_noise in unique_obs_img_noise:
    #     reduced_uniques.append(reduce_resolution(uniq_noise))

    #unique_obs_img_preproc, orig_max, orig_min = preprocess(reduced_uniques, histo_bins=24)
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(reduced_uniques, mean=None, std=None)

    longest_plan_transis_test = []

    test_noisy_transis = []

    # main loop 
    for iii, trace in enumerate(loaded_traces):


        nb_additional_noisy_trans = 0

        actions_for_one_trace = []

        for trtrtr, transitio in enumerate(trace):


            # initttt = "300000000001000000000000000000000000",
            # goallll = "300001000000000000000000000000000000",
            # if replace_in_str(transitio[1])[0] == "300000000001000000000000000000000000":
            #     print("salut")
            #     print(replace_in_str(transitio[1]))
            #     #exit()
            # continue

            if remove_a_trans:

                if replace_in_str(transitio[1])[0] in trans_to_remove_list and replace_in_str(transitio[1])[1] in trans_to_remove_list:
                    
                    continue

            # normalize the two images
            #transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0]), histo_bins=24)
            transi_0_reduced_and_norm, _, _ =   normalize_colors(reduce_resolution(transitio[0][0]), mean=mean_all, std=std_all)

            #transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1]), histo_bins=24)
            transi_1_reduced_and_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][1]), mean=mean_all, std=std_all)


            # exit()
        
            # if transition was not looped over, add id to the "unique" arrays AND 
            # create the init/goal (non noisy) if needed
            if str(transitio[1][3]) not in all_actions_unique:

                # add it to all_actions_unique
                all_actions_unique.append(str(replace_in_str(transitio[1])))

                # add it to all_transitions_unique
                all_transitions_unique.append([replace_in_str(transitio[1])[0], replace_in_str(transitio[1])[1]])

                #### SAVING THE INIT/GOAL IMAGES (non noisy version)
                if create_init_goal:
                    # print("exp_folderexp_folder")
                    # print(exp_folder)
                    # exit()


                    if trtrtr % 50 == 0:
                        save_np_image(exp_folder, transi_0_reduced_and_norm, "pair_"+str(trtrtr)+"_0")
                        save_np_image(exp_folder, transi_1_reduced_and_norm, "pair_"+str(trtrtr)+"_1")

                    if replace_in_str(transitio[1])[0] == init_st.replace("_noise", ""):
                        save_np_image(exp_folder, transi_0_reduced_and_norm, "init_NoNoise")

                        # im2 = unnormalize_colors(transi_0_reduced_and_norm, mean_all, std_all)      
                        # plt.imsave("SOK_INITT-val.png", im2)
                        # plt.close()
                        #exit()

                    if replace_in_str(transitio[1])[1] == init_st.replace("_noise", ""):
                        save_np_image(exp_folder, transi_1_reduced_and_norm, "init_NoNoise")
                        # plt.imsave("HANOI_INIT-val.png", reduce_resolution(transitio[0][1]))


                        im2 = unnormalize_colors(transi_1_reduced_and_norm, mean_all, std_all)      
                        plt.imsave("SOK_INITT-val.png", im2)
                        plt.close()

                    if replace_in_str(transitio[1])[0] == goal_st.replace("_noise", ""):
                        save_np_image(exp_folder, transi_0_reduced_and_norm, "goal_NoNoise")
                    
                        im2 = unnormalize_colors(transi_0_reduced_and_norm, mean_all, std_all)
                        #im2 = np.clip(im2, 0, 1)
                        plt.imsave("SOK_GOAL-val.png", im2)
                        plt.close()

                    if replace_in_str(transitio[1])[1] == goal_st.replace("_noise", ""):
                        save_np_image(exp_folder, transi_1_reduced_and_norm, "goal_NoNoise")

                        im2 = unnormalize_colors(transi_1_reduced_and_norm, mean_all, std_all)
                        #im2 = np.clip(im2, 0, 1)
                        plt.imsave("SOK_GOAL-val.png", im2)
                        plt.close()



            # If the current transition is not in the NOISY SET
            if str(replace_in_str(transitio[1])) not in all_the_unique_actions_noisy_part_count:

                all_images_reduced_and_norm.append(transi_0_reduced_and_norm)
                all_images_reduced_and_norm.append(transi_1_reduced_and_norm)
                
                actions_for_one_trace.append(str(replace_in_str(transitio[1])))


                # ADDING THE IMAGES to a unique set
                if str(replace_in_str(transitio[1])[0]) not in all_images_reduced_and_norm_uniques_nonnoisy:
                    all_images_reduced_and_norm_uniques_nonnoisy[str(replace_in_str(transitio[1])[0])] = transi_0_reduced_and_norm

                if str(replace_in_str(transitio[1])[1]) not in all_images_reduced_and_norm_uniques_nonnoisy:
                    all_images_reduced_and_norm_uniques_nonnoisy[str(replace_in_str(transitio[1])[1])] = transi_1_reduced_and_norm


            # NOISY PART
            elif add_noisy_trans:


                # update the count
                all_the_unique_actions_noisy_part_count[str(replace_in_str(transitio[1]))] += 1
                
                # produce number_noisy_versions versions of the transitons

                for inndex in range(number_noisy_versions):

                    nb_additional_noisy_trans += 1


                    if 'preprocess' in noise_type:
                        #transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0]), histo_bins=24)
                        transi_0_prepro_img_color_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][0]), mean=mean_all, std=std_all)
                        random.seed(inndex)
                        np.random.seed(inndex)
                        gaussian_noise_0 = np.random.normal(mean, std, transi_0_prepro_img_color_norm.shape)
                        random.seed(1)
                        np.random.seed(1)
                        noisy1_reduced_preproc_norm = transi_0_prepro_img_color_norm + gaussian_noise_0
                        noisy1 = noisy1_reduced_preproc_norm

                        #transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1]), histo_bins=24)
                        transi_1_prepro_img_color_norm, _, _ = normalize_colors(reduce_resolution(transitio[0][1]), mean=mean_all, std=std_all)
                        random.seed(inndex)
                        np.random.seed(inndex)
                        gaussian_noise_1 = np.random.normal(mean, std, transi_1_prepro_img_color_norm.shape)
                        random.seed(1)
                        np.random.seed(1)
                        noisy2_reduced_preproc_norm = transi_1_prepro_img_color_norm + gaussian_noise_1
                        noisy2 = noisy2_reduced_preproc_norm

                        #noisy1 = np.clip(noisy2, 0, 255).astype(np.uint8)              

                    if replace_in_str(transitio[1])[0] not in all_images_reduced_and_norm_uniques_noisy:
                        all_images_reduced_and_norm_uniques_noisy[replace_in_str(transitio[1])[0]] = noisy1_reduced_preproc_norm

                    if replace_in_str(transitio[1])[1] not in all_images_reduced_and_norm_uniques_noisy:
                        all_images_reduced_and_norm_uniques_noisy[replace_in_str(transitio[1])[1]] = noisy2_reduced_preproc_norm
                    
                    # if 'noise' init/state was chosen
                    # AND if one of the present states is init/state
                    # save the colored normalized array in some picke file
                    print("laaaa1")
                    if create_init_goal:
                        print("laaaa2")
                        #exit()
                        #if 'noise' in init_st:
                        if replace_in_str(transitio[1])[0] == init_st.replace("_noise", ""):
                            save_np_image(exp_folder_, noisy1, "init-noise-"+str(inndex))

                        if replace_in_str(transitio[1])[1] == init_st.replace("_noise", ""):
                            save_np_image(exp_folder_, noisy2, "init-noise-"+str(inndex))

                        #if 'noise' in goal_st:

                        if replace_in_str(transitio[1])[0] == goal_st.replace("_noise", ""):
                            # print("laaaaa")
                            # exit()
                            save_np_image(exp_folder_, noisy1, "goal-noise-"+str(inndex))
                            # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(noisy1))
                            # plt.close()

                        if replace_in_str(transitio[1])[1] == goal_st.replace("_noise", ""):
                            # print("laaaaa1111111111")
                            # exit()
                            save_np_image(exp_folder_, noisy2, "goal-noise-"+str(inndex))
                            # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(noisy2))
                            # plt.close()


                    # fill in the all_images_reduced_and_norm with the noisy images
                    all_images_reduced_and_norm.append(noisy1_reduced_preproc_norm)
                    all_images_reduced_and_norm.append(noisy2_reduced_preproc_norm)

                    # fill in the array of actions of the current trace
                    #actions_for_one_trace.append(str(replace_in_str(transitio[1])))

                    # If the trans labels must be differentiated !
                    if str(replace_in_str(transitio[1])) in all_the_unique_actions_ten_percent_noisy_and_dup:
                        actions_for_one_trace.append(str(replace_in_str(transitio[1]))+"_"+str(inndex))

                        # ADD the action to the unique vectors
                        if str(replace_in_str(transitio[1]))+"_"+str(inndex) not in all_actions_unique:
                            all_actions_unique.append(str(replace_in_str(transitio[1]))+"_"+str(inndex))
                            #all_transitions_unique.append([part1_trans_+"_"+str(inndex), part2_trans_+"_"+str(inndex)])
                            all_transitions_unique.append([replace_in_str(transitio[1])[0]+"_"+str(inndex), replace_in_str(transitio[1])[1]+"_"+str(inndex)])
                    # or notall_transitions_unique

                    else:
                        actions_for_one_trace.append(str(replace_in_str(transitio[1])))
                        

        # add the actions of this trace to all_actions_for_trace (the array of all actions of all traces)
        all_actions_for_trace.append(actions_for_one_trace)

        # add the indices for this trace to traces_indices array
        traces_indices.append([start_trace_index, start_trace_index+(len(trace)+nb_additional_noisy_trans)*2])
        
        # update the index for the next trace
        start_trace_index+=(len(trace)+nb_additional_noisy_trans)*2

    if not os.path.exists("all_states_noisy_sokoban"):
        os.makedirs("all_states_noisy_sokoban") 
    exp_folder_all_states = "all_states_noisy_sokoban"      

    for k, v in all_images_reduced_and_norm_uniques_nonnoisy.items():


        # unorm = unnormalize_colors(v, mean_all, std_all) 
        # dehanced = deenhance(unorm)
        # denormalized = denormalize(dehanced, orig_min, orig_max)
        # denormalized = np.clip(denormalized, 0, 1)
        # plt.imsave(exp_folder_all_states+"/image"+str(k)+".png", denormalized)
        # plt.close()

        save_np_image(exp_folder_all_states, v, "image"+str(k))  


    for k, v in all_images_reduced_and_norm_uniques_noisy.items():
        # print("SHOULD NOT BE HERE")
        # exit()
        save_np_image(exp_folder_all_states, v, "image"+str(k)+"noisy") 


    # print(longest_plan_transis_test)

    # print("len(all_images_reduced)") # 3194
    # print(len(all_images_reduced))
    # print(test_noisy_transis)
    # exit()
    print("laaa")
    if build_dfa:

        ############################################
        ######## Building the file for the DFA #####
        ############################################

        # Input:
        # all_actions_unique
        # 



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


        # print(replace_in_str(transitio[1]))
        
        print("iici")
        init_st = "400000001200000000000000000000000000"
        goal_st = "400000001200000000000000000000000000"

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

        with open("Full_DFA_Sokoban_6_6.json", 'w') as f:
            json.dump(total_dfa_dico, f)

        ############################################
        #### END Building the file for the DFA #####
        ############################################


    # with open("all_actionsPRE.txt","w") as f:
    #     for i in range(len(all_actions_unique)):
    #         #np.argmax(all_actions_one_hot[i]
    #         f.write("a"+str(i)+" is "+str(all_actions_unique[i])+"\n")

    # exit()


    # si trans parcouru == "[{"+part2_trans+"}, {"+part1_trans+"}]"



    # all_images_reduced > gaussian > clip > preprocess > normalize_colors

   
    # counntteerr=0
    # all_images_reduced_gaussian_20 = add_noise(all_images_reduced, seed=1)
    # counntteerr=0
    # all_images_reduced_gaussian_30 = add_noise(all_images_reduced, seed=2)
    # counntteerr=0
    # all_images_reduced_gaussian_40 = add_noise(all_images_reduced, seed=3)

    # save_noisy("sokoban_5_5_dataset", "all_images_seed1.p", all_images_reduced_gaussian_20)
    # save_noisy("sokoban_5_5_dataset", "all_images_seed2.p", all_images_reduced_gaussian_30)
    # save_noisy("sokoban_5_5_dataset", "all_images_seed3.p", all_images_reduced_gaussian_40)


    # loaded_noisy1 = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban_5_5_dataset/all_images_seed1.p")
    # all_images_seed1 = loaded_noisy1["images"]

    # loaded_noisy2 = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban_5_5_dataset/all_images_seed2.p")
    # all_images_seed2 = loaded_noisy2["images"]

    # loaded_noisy3 = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban_5_5_dataset/all_images_seed3.p")
    # all_images_seed3 = loaded_noisy3["images"]

    


    # all_images_reduced_gaussian_20_preproc, _, _ = preprocess(all_images_reduced, histo_bins=256)
    # all_images_reduced_gaussian_20_norm, __, __ = normalize_colors(all_images_reduced_gaussian_20_preproc, mean=mean_all, std=std_all, second=True)


    # all_images_reduced_gaussian_30_preproc, _, _ = preprocess(all_images_reduced, histo_bins=256)
    # all_images_reduced_gaussian_30_norm, __, __ = normalize_colors(all_images_reduced_gaussian_30_preproc, mean=mean_all, std=std_all)

    # all_images_reduced_gaussian_40_preproc, _, _ = preprocess(all_images_reduced, histo_bins=256)
    # all_images_reduced_gaussian_40_norm, __, __ = normalize_colors(all_images_reduced_gaussian_40_preproc, mean=mean_all, std=std_all)

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
    
    orig_max, orig_min = None, None
    

    # 4115
    print("len train_set")
    print(len(train_set[0][1]))

    return train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min





train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()


# print("aaaaa")
# print(len(train_set[0][1]))

# exit()




# for hh in range(0, len(train_set), 200):

#     acc = all_actions_unique[np.argmax(train_set[hh][1])]
#     print("action for {} is {}".format(str(hh), str(acc)))


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


#     plt.imsave("sokoban-5-5-pair_"+str(hh)+"_pre.png", im1)
#     plt.close()

#     plt.imsave("sokoban-5-5-pair_"+str(hh)+"_suc.png", im2)
#     plt.close()


# exit()


