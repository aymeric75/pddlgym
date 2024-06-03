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
from skimage import exposure, color
import itertools
from PIL import Image

#from networkX_returnTransitionsToRemove import return_transitions_to_remove


import argparse
parser = argparse.ArgumentParser(description="A script to generate traces from a domain")
parser.add_argument('domain', type=str, help='domain name')
parser.add_argument('traces_dir', type=str, help='absolute dir where the .p file of the trace will be')

args = parser.parse_args()

random.seed(1)
np.random.seed(1)



##################### HANOI #######################

def origin_and_destination(action, peg_to_disc_list_pre, peg_to_disc_list_suc):

    # take the sent disk name
        
    disk_name = action.variables[0].name

    pre_peg = ''
    suc_peg = ''

    # look where it is in the pre lists
    for index, dico in enumerate(peg_to_disc_list_pre.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                pre_peg = "peg"+str(index+1)

    # look where it is in the suc list
    for index, dico in enumerate(peg_to_disc_list_suc.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                suc_peg = "peg"+str(index+1)

    # return the two "where"
    return pre_peg, suc_peg

def generate_hanoi_traces():

    nb_samplings_per_starting_state = 501
    ##### requirements for each datasets

    #### string for the .make fct
    ####

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi44-v0", dynamic_action_space=True)

    all_traces = []
    unique_obs = []
    unique_obs_img = []
    unique_transitions = [] # each ele holds a str representing a unique pair of peg_to_disc_list
    obs_occurences = {}
    counter = 0

    # looping over the number of starting positions
    for ii in range(1, 52, 1):

        last_two_peg_to_disc_lists_str = [] # must contain only two lists that represent a legal transition
        last_two_peg_to_disc_lists = []
        last_two_imgs = []
        trace_transitions = []

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        # Retrieve the 1st image
        img, peg_to_disc_list = env.render()
        img = img[:,:,:3] # remove the transparancy

        if str(peg_to_disc_list) not in unique_obs:
            unique_obs.append(str(peg_to_disc_list))
            obs_occurences[str(peg_to_disc_list)] = 1
            unique_obs_img.append(img)

        last_two_peg_to_disc_lists_str.append(str(peg_to_disc_list))
        last_two_peg_to_disc_lists.append(peg_to_disc_list)
        last_two_imgs.append(img)

        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_per_starting_state):
            
            if counter%10 == 0:
                print("counter: {}".format(str(counter)))

            # sample an action
            action = env.action_space.sample(obs)

            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)
            img, peg_to_disc_list = env.render()
            img = img[:,:,:3]

            last_two_peg_to_disc_lists_str.append(str(peg_to_disc_list))
            last_two_peg_to_disc_lists.append(peg_to_disc_list)
            if len(last_two_peg_to_disc_lists_str) > 2:
                last_two_peg_to_disc_lists_str.pop(0)
                last_two_peg_to_disc_lists.pop(0)

            last_two_imgs.append(img)
            if len(last_two_imgs) > 2:
                last_two_imgs.pop(0)

            if len(last_two_peg_to_disc_lists_str) == 2:

                if str(last_two_peg_to_disc_lists_str) not in unique_transitions and str(last_two_peg_to_disc_lists[0]) != str(last_two_peg_to_disc_lists[1]):

                    transition_actions = [] # hold all the version of the action
                    # i.e. loose, semi-loose-v1, semi-loose-v2

                    pre_peg, post_peg = origin_and_destination(action, last_two_peg_to_disc_lists[0], last_two_peg_to_disc_lists[1])

                    transition_actions.append(str(action))
                    transition_actions.append(str(action)+pre_peg+post_peg)
                    transition_actions.append(str(action)+pre_peg)
                    transition_actions.append(str(last_two_peg_to_disc_lists)) # action full description
                    unique_transitions.append(str(last_two_peg_to_disc_lists_str))
                    trace_transitions.append([[last_two_imgs[0], last_two_imgs[1]], transition_actions])

            if str(peg_to_disc_list) not in unique_obs:
                unique_obs.append(str(peg_to_disc_list))
                obs_occurences[str(peg_to_disc_list)] = 1
                unique_obs_img.append(img)
            else:
                obs_occurences[str(peg_to_disc_list)] += 1

            counter += 1

        all_traces.append(trace_transitions)

    print("number of unique transitions is : {}".format(str(len(unique_transitions))))

    with open("resultatHanoi4-4.txt", 'w') as file2:

        file2.write(str(len(unique_transitions)) + '\n')

    return all_traces, obs_occurences, unique_obs_img, unique_transitions


##################### BLOCKS #######################

def generate_blocks_traces():

    nb_samplings_per_starting_state = 501

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvBlocks4colors-v0", dynamic_action_space=True)

    all_traces = []
    unique_obs = []
    unique_obs_img = []
    unique_transitions = [] # each ele holds a str representing a unique pair of blocks_data
    obs_occurences = {}
    counter = 0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    # 19 !!!!
    for ii in range(0, 19, 1):

        print("ici")

        last_two_blocks_data_str = [] # must contain only two lists that represent a legal transition
        last_two_blocks_data = []
        last_two_imgs = []
        trace_transitions = []

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        # Retrieve the 1st image
        img, blocks_data = env.render()

        # exit()
        # if str(blocks_data) == "([[], [], [], [d:block, a:block, b:block, c:block]], None)":
        #     print("okkk")
        #     print(blocks_data)
        #     exit()

        img = img[:,:,:3] # remove the transparancy




        # gray_image = np.mean(img, axis=2)
        # threshold = 127  # This is a common default value, but you may need to adjust it
        # binary_image = np.where(gray_image > threshold, 255, 0)
        # img = np.stack([binary_image]*3, axis=-1)
    
        #print(img.shape)

        #print(img[:,:,0].shape)
        #print(np.where(img[:,:,0] > 0, 255, 0))
        #img = np.where(img[:,:,0] > 0, 1, 0)
        #img = reduce_resolution(img)
        
        

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



            #img = np.where(img[:,:,0] > 0, 1, 0)
            # gray_image = np.mean(img, axis=2)
            # threshold = 127  # This is a common default value, but you may need to adjust it
            # binary_image = np.where(gray_image > threshold, 255, 0)
            # img = np.stack([binary_image]*3, axis=-1)

            last_two_blocks_data_str.append(str(blocks_data))
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


                    #transition_actions.append(str(action))
        
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
    print("number of unique_obs_img is : {}".format(str(len(unique_obs_img))))


    with open("resultatBlocks4.txt", 'w') as file2:

        file2.write(str(unique_transitions) + '\n')

    return all_traces, obs_occurences, unique_obs_img, unique_transitions


##################### SOKOBAN #######################

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

def find_index_2d(array, target):
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            if value == target:
                return (i, j)
    return None  # 

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

def generate_all_transitions():

    all_transitions = []


    for state in generate_all_states():

        next_states = generate_next_states(state)


        for next_ in next_states:


            if str(find_index_2d(next_, 2)) == str(find_index_2d(state, 1)):
                continue


            all_transitions.append([state, next_])

    return all_transitions

def generate_sokoban_traces(transitions):

    nb_samplings_per_starting_state = 501 #3001

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



##################### SAVE AND LOAD DATASET #######################

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
    filename = "traces.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)



print("args.traces_dir tttt")
print(args.traces_dir)

if args.domain == "hanoi":
    all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_hanoi_traces()
    save_dataset(args.traces_dir, all_traces, obs_occurences, unique_obs_img, unique_transitions)
elif args.domain == "blocks":
    all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_blocks_traces()
    save_dataset(args.traces_dir, all_traces, obs_occurences, unique_obs_img, unique_transitions)
elif args.domain == "sokoban":
    all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_sokoban_traces(generate_all_transitions())
    save_dataset(args.traces_dir, all_traces, obs_occurences, unique_obs_img, unique_transitions)




