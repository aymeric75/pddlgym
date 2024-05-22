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


random.seed(1)
np.random.seed(1)

all_the_indx = []

def all_colors():
    #colors = [0, 64, 128, 192, 255]
    colors = [0, 128, 255]
    combinations = list(itertools.product(colors, repeat=3))
    return combinations

# Define a function to convert an RGB combination to LAB
def convert_to_lab(rgb):
    # skimage expects RGB values to be normalized between 0 and 1
    normalized_rgb = [value / 255.0 for value in rgb]
    # Reshape the RGB tuple to a 1x1x3 numpy array as required by rgb2lab
    rgb_array = np.array([normalized_rgb]).reshape((1, 1, 3))
    lab_array = color.rgb2lab(rgb_array)
    # Return the LAB value, flattened back into a simple tuple
    return tuple(lab_array.flatten())


lab_combinations = list(map(convert_to_lab, all_colors()))

counntteerr = 0


ref_colors = [
    [ 0 ,  0, 255],
    [  0 ,255 ,0],
    [  0 ,255 ,  0],
    [128  ,128 ,128],
    [255, 0, 0],
    [255 ,255  , 0],
    [128, 255, 255]]


shades = [[128, 0, 255],
    [128, 255, 0],
    [0, 128, 0],
    [255, 128, 128],
    [128, 0, 0],
     [128, 255, 0],
     [128, 128, 255]]


def convert_to_lab_and_to_color_wt_min_distance(rgb, boolean_matrix):

    global counntteerr

    if counntteerr % 100000 == 0:
        print("counter is {}".format(str(counntteerr)))

    if counntteerr == len(boolean_matrix):
        return rgb

    #if counntteerr < len(boolean_matrix):
    if boolean_matrix[counntteerr]:



        if (rgb == np.array(ref_colors[0])).all():
            counntteerr+=1
            return np.array(shades[0])

        elif (rgb == np.array(ref_colors[1])).all():
            counntteerr+=1
            return np.array(shades[1])

        elif (rgb == np.array(ref_colors[2])).all():
            counntteerr+=1
            return np.array(shades[2])

        elif (rgb == np.array(ref_colors[3])).all():
            counntteerr+=1
            return np.array(shades[3])

        elif (rgb == np.array(ref_colors[4])).all():
            counntteerr+=1
            return np.array(shades[4])

        elif (rgb == np.array(ref_colors[5])).all():
            counntteerr+=1
            return np.array(shades[5])

        elif (rgb == np.array(ref_colors[6])).all():
            counntteerr+=1
            return np.array(shades[6])




        lab_array = color.rgb2lab(rgb)
        all_dists = all_distances(lab_array)
        # put the "0" dist to infinit (coz we dont want to spot this one)
        if 0 in all_dists:
            all_dists[all_dists.index(0)] = 99999
        # retrieve the index of the min distance (expect when is equal)
        index_min_color = np.argmin(all_dists)
        closest_color = all_colors()[index_min_color]
        counntteerr+=1
        return closest_color
    
    else:
        counntteerr+=1
        return rgb



def all_distances(labcolor):

    all_distances_lab = list(map(lambda x: color.deltaE_cie76(labcolor, x), lab_combinations))
    #all_distances_lab = list(map(lambda x: color.deltaE_ciede2000(labcolor, x), lab_combinations))
    #all_distances_lab = list(map(lambda x: color.deltaE_ciede94(labcolor, x), lab_combinations))

    return all_distances_lab



def add_noise(images, seed=0):

    np.random.seed(seed)

    # Reshape the array to have each row represent a pixel's color
    pixels = images[0].reshape(-1, 3)
    # Find unique color combinations
    unique_colors = np.unique(pixels, axis=0)
    print("luu")
    print(unique_colors)
    #exit()

    if not isinstance(images, np.ndarray):
        images = np.array(images)

    # print(images.shape) # (768, 25, 70, 3)
    # # boolean_matrix 



    # convert the image dataset into a "lab" format 

    reshaped_rgb = images.reshape(-1, 3)  # Reshape to a 2D array where each row is a pixel's RGB values
    print(reshaped_rgb.shape) # 5 082 000


    # Generate a matrix of random numbers from a uniform distribution
    random_matrix = np.random.uniform(low=0.0, high=1.0, size=(reshaped_rgb.shape[0],))
    # Create a boolean matrix: True if the element is < 0.05, False otherwise
    boolean_matrix = random_matrix < 0.01
    #print(boolean_matrix)

    lab_pixels = np.apply_along_axis(convert_to_lab_and_to_color_wt_min_distance, 1, reshaped_rgb, boolean_matrix)
    


    #print("seed {}, images.shape {}, reshaped_rgb {}, lab_pixels {}, bool matrix {}".format(str(seed), str(images.shape), str(reshaped_rgb.shape), str(lab_pixels.shape), str(boolean_matrix.shape)))


    #dataset_lab = lab_pixels.reshape(768, 25, 70, 3)
    dataset_lab = lab_pixels.reshape(images.shape)

    # NOW holds the closests values  

    return dataset_lab


def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min()), image.max(), image.min()

def equalize(image, histo_bins=None):
    if image.shape == (25, 70, 3):
        return exposure.equalize_hist(image, mask=np.ones(image.shape))
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image, histo_bins=256):

    if not isinstance(image,np.ndarray):
        image = np.array(image)

    # compte le nombre de couleurs
    # pour chacune, tu donne une version modifiée (voir l'autre fct là)
    # pour le nbre de bins, bah c'est le nbre de couleurs
    # print("debut")
    # print(image.shape)
    image = image / 255. # put the image to between 0 and 1 (with the assumption that vals are 0-255)
    #print(np.unique(image))
    # [0.         0.00392157 0.00784314 ... 1.]
    image = image.astype(float)
    #
    image = equalize(image, histo_bins=14)
    #print(np.unique(image))
    # [0.0579892  0.05867434 0.05999787 ... 1.]
    image, orig_max, orig_min = normalize(image) # put the image back to btween 0 and 1
    #print(np.unique(image))
    # [0.00000000e+00 7.27317562e-04 2.13232562e-03 .... 1.]
    image = enhance(image) # i) from btween 0-1 values, center to 0 (so become btween -0.5 +0.5)
    #print(np.unique(image))
    # [0.         0.80720517 1.        ]


    # gaussian noise of N(0, 0.2) on 1 === ?

    # gaussian noise on 255 === ?


    return image, orig_max, orig_min



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



def normalize_colors(images, mean=None, std=None, second=False):    

    if mean is None or std is None:

        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)

    return (images - mean)/(std+1e-20), mean, std




# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):

    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                            np.arange(0, image.shape[1], 9), axis=1)

    return reduced_image


def unnormalize_colors(normalized_images, mean, std): 
    # Reverse the normalization process
    unnormalized_images = normalized_images * (std + 1e-6) + mean
    return np.round(unnormalized_images).astype(np.uint8)



# return the origin and destination of a moved disk
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


# 1401
nb_samplings_per_starting_state = 501 # has to be ODD 




def generate_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi44-v0", dynamic_action_space=True)

    all_traces = []
    unique_obs = []
    unique_obs_img = []
    unique_transitions = [] # each ele holds a str representing a unique pair of peg_to_disc_list
    obs_occurences = {}
    counter = 0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
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



# construct pairs (of images)
# and construct the array of action (for each pair of images) 
# Return: pairs of images, corresponding actions (one-hot)

def modify_one_trace(all_images_transfo_tr, all_images_orig_tr, all_actions, all_actions_unique_):

    #print(type(all_images_transfo_tr))
    # building the array of pairs
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    for iiii, p in enumerate(all_images_transfo_tr):
        if iiii%2 == 0:


            # mask = all_images_transfo_tr[iiii+1] > 1000
            # result = all_images_transfo_tr[iiii+1][mask]
            # if len(result) > 0:
            #     print(result)
            #     exit()

            #if iiii < len(all_images_transfo_tr)-1:
            all_pairs_of_images.append([all_images_transfo_tr[iiii], all_images_transfo_tr[iiii+1]])
            #all_pairs_of_images_orig.append([all_images_orig_tr[iiii], all_images_orig_tr[iiii+1]])

    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions:
        

        # if 'noise' in ac:
        #     print("jai trouveeee")
        #     print(ac)
        #     print(all_actions_unique_.index(str(ac)))
        if all_actions_unique_.index(str(ac)) not in all_the_indx:
            all_the_indx.append(all_actions_unique_.index(str(ac)))

        # if all_actions_unique_.index(str(ac)) > 1595:
        #     print("ctoutvu")
        #     exit()


        all_actions_indices.append(all_actions_unique_.index(str(ac)))
    

    # print("ouaicgh")
    # print(len(all_actions_unique_)) # 1597

    actions_indexes = torch.tensor(all_actions_indices).long()

    # array of one hot encoded actions
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique_))

    # print("aaaaaa")
    # print(len(actions_one_hot)) # normalement 1597
    #exit()

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


def save_np_image(dire, array, file_name):
    data = {
        "image": array,
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = str(file_name)+".p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)


def save_noisy(dire, filename, images):
    data = {
        "images": images
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)



# # 1) generate dataset (only once normally)
# all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_dataset()

# # # 2) save dataset
# # save_dataset("hanoi_4_4_dataset", all_traces, obs_occurences, unique_obs_img, unique_transitions)

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


import re

# Regular expression to find the pattern [LETTERNUMBER]
pattern = r"\[([a-zA-Z])([0-9])\]"

# Function to replace the pattern [LETTERNUMBER] with LETTERNUMBER
def replace_pattern(match):
    return f"{match.group(1)}{match.group(2)}"


def replace_in_str(word):
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

        #res1 = replace_pattern(res1, pattern)
        return res1



def rgb_to_hsv_with_pil(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    hsv_image = image.convert('HSV')
    #hsv_array = np.array(hsv_image)
    return hsv_image



def hsv_to_rgb_with_pil(image):
    image = Image.fromarray(image.astype('uint8'), 'HSV')
    rgb_image = image.convert(mode="RGB")
    return rgb_image


def add_gaussian_noise_to_hsv(hsv_img, mean=0, std_dev=15, hsv_channel="h"):

    
    # Extract the channels as numpy arrays
    h, s, v = hsv_img.split()

    chan = None

    if hsv_channel == "h": chan = h
    elif hsv_channel == "s": chan = s
    elif hsv_channel == "v": chan = v

    chan = np.array(chan, dtype=np.int16)  # Convert to int16 for safe addition of noise
    
    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, chan.shape).astype(np.int16)
    chan += noise  # Add noise to the hue channel
    np.clip(chan, 0, 255, out=chan)  # Clip values to keep them within the 0-255 range
    chan = chan.astype(np.uint8)
    
    # Merge the channels back and convert to RGB
    modified_hsv_img = None

    if hsv_channel == "h": modified_hsv_img = Image.merge('HSV', (Image.fromarray(chan), s, v))
    elif hsv_channel == "s": modified_hsv_img = Image.merge('HSV', (h, Image.fromarray(chan), v))
    elif hsv_channel == "v": modified_hsv_img = Image.merge('HSV', (h, s, Image.fromarray(chan)))
    
    
    return modified_hsv_img



def plot_grid(images,w=10,path="plan.png",verbose=False):
    import matplotlib.pyplot as plt
    import math
    l = 0
    l = len(images)
    h = int(math.ceil(l/w))
    plt.figure(figsize=(w*1.5, h*1.5))
    for i,image in enumerate(images):
        ax = plt.subplot(h,w,i+1)
        try:
            plt.imshow(image,interpolation='nearest',cmap='gray', vmin = 0, vmax = 1)
        except TypeError:
            TypeError("Invalid dimensions for image data: image={}".format(np.array(image).shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print(path) if verbose else None
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# to specify that you want the noisy version for init/goal, add _noise

def export_dataset(action_type="full",
        remove_a_trans=False,
        remove_some_trans=True,
        build_dfa=False,
        add_noisy_trans=True,
        ten_percent_noisy_and_dupplicated=False,
        noise_type='preprocess',
        number_noisy_versions=3,
        init_st = "[d4d2]d1d3E",
        goal_st = "d4d1d3d2",
        create_init_goal=True,
        perc=55,
        std=0.015    #0.015
        ):



    # 'Ed4E[d3d2d1]', 'Ed4d1[d3d2]', 'E[d4d2]d1d3', 'd3[d4d2]d1E', '[d3d2]d4d1E', '[d3d2]Ed1d4', 'd3d2d1d4', 'Ed2d1[d4d3]', 'EEd1[d4d3d2]', 'EEE[d4d3d2d1]'

    exp_folder = None
    exp_folder_noisy = None
    exp_folder_partial = None
    exp_folder_ten_percent_noisy_and_dupplicated = None



    if not os.path.exists("Hanoi_4_4_FullDFA"+init_st+"-"+goal_st):
        os.makedirs("Hanoi_4_4_FullDFA"+init_st+"-"+goal_st) 
    exp_folder = "Hanoi_4_4_FullDFA"+init_st+"-"+goal_st

    global counntteerr

    trans_to_remove_list = ['[d4d2]d1d3E', 'd4d1d3d2']

    transis_to_remove = []
    if remove_some_trans:
        loaded_removable_trans = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_missing_edges_2/deleted_edges.p")
        transis_to_remove = loaded_removable_trans["stuff"]

        loaded_removable_edge_goal = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_missing_edges_2/goal_state.p")
        edge_removed_goal = loaded_removable_edge_goal["stuff"]
        goal_st = edge_removed_goal

        loaded_removable_edge_init = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_missing_edges_2/init_state.p")
        edge_removed_init = loaded_removable_edge_init["stuff"]
        init_st = edge_removed_init

    
    transis_to_remove_str = []
    for tr in transis_to_remove:
        transis_to_remove_str.append(str(tr))

    print("len transis_to_remove_str {}".format(str(len(transis_to_remove_str)))) # 519


    # if not os.path.exists("hanoi_4_4_FullDFA-no-noise-"+init_st+"-"+goal_st):
    #     os.makedirs("hanoi_4_4_FullDFA-no-noise-"+init_st+"-"+goal_st) 
    # exp_folder_ = "hanoi_4_4_FullDFA-no-noise-"+init_st+"-"+goal_st


    if not os.path.exists("hanoi_4_4_FullDFA-with-noise"+init_st+"-"+goal_st):
        os.makedirs("hanoi_4_4_FullDFA-with-noise"+init_st+"-"+goal_st) 
    exp_folder_ = "hanoi_4_4_FullDFA-with-noise"+init_st+"-"+goal_st

    if add_noisy_trans:
        # creating init/goal dir for the experiment
        if not os.path.exists("hanoi_4_4_perc"+str(perc)+"_#v"+str(number_noisy_versions)+"_std"+str(std)):
            os.makedirs("hanoi_4_4_perc"+str(perc)+"_#v"+str(number_noisy_versions)+"_std"+str(std)) 
        exp_folder_noisy = "hanoi_4_4_perc"+str(perc)+"_#v"+str(number_noisy_versions)+"_std"+str(std)

    if remove_a_trans:
        # creating init/goal dir for the experiment
        if not os.path.exists("hanoi_4_4_RemovedTrans-"+trans_to_remove_list[0]+"-"+trans_to_remove_list[1]):
            os.makedirs("hanoi_4_4_RemovedTrans-"+trans_to_remove_list[0]+"-"+trans_to_remove_list[1]) 
        exp_folder = "hanoi_4_4_RemovedTrans-"+trans_to_remove_list[0]+"-"+trans_to_remove_list[1]

    if remove_some_trans:
        if not os.path.exists("hanoi_4_4_missing_edges"):
            os.makedirs("hanoi_4_4_missing_edges") 
        exp_folder_partial = "hanoi_4_4_missing_edges"

    if ten_percent_noisy_and_dupplicated:
        if not os.path.exists("ten_percent_noisy_and_dupplicated"):
            os.makedirs("ten_percent_noisy_and_dupplicated") 
        exp_folder_ten_percent_noisy_and_dupplicated = "ten_percent_noisy_and_dupplicated"      

    if not os.path.exists("all_states"):
        os.makedirs("all_states") 
    exp_folder_all_states = "all_states"      


    # mean and std for the noise
    mean = 0

    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/data.p")


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

    print("BEFORE")

    print(len(transis_to_remove_str)) # 519 



    if remove_some_trans:
        loaded_traces_after_prunning = []
        for iii, trace in enumerate(loaded_traces):
            #trace_copy = trace.copy()
            if len(trace) == 0:
               continue
            
            trace_tmp = []
            for trtrtr, transitio in enumerate(trace):
                first_split_ = transitio[1][3].split("}, {")
                part1_trans_ = replace_in_str(first_split_[0].replace('[{', ''))
                part2_trans_ = replace_in_str(first_split_[1].replace('}]', ''))

                if str([part1_trans_, part2_trans_]) not in transis_to_remove_str:
                    trace_tmp.append(transitio)

            #print("len trace_tmp {}".format(str(len(trace_tmp))))

            loaded_traces_after_prunning.append(trace_tmp)       

        loaded_traces = loaded_traces_after_prunning




    ## first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_traces):
        for trtrtr, transitio in enumerate(trace):
            if str(transitio[1][3]) not in all_the_unique_actions:

                first_split_ = transitio[1][3].split("}, {")
                part1_trans_ = replace_in_str(first_split_[0].replace('[{', ''))
                part2_trans_ = replace_in_str(first_split_[1].replace('}]', ''))
                
                
                if replace_in_str(part1_trans_) == "d4[d2d1]Ed3":
                    plt.imsave("HANOI_INIT_EX.png", transitio[0][0])
                    plt.close()      
                if replace_in_str(part2_trans_) == "d4[d2d1]Ed3":
                    plt.imsave("HANOI_INIT_EX.png", transitio[0][1])
                    plt.close()
                if replace_in_str(part1_trans_) == "[d4d3][d2d1]EE":
                    plt.imsave("HANOI_GOAL_EX.png", transitio[0][0])
                    plt.close() 
                if replace_in_str(part2_trans_) == "[d4d3][d2d1]EE":
                    plt.imsave("HANOI_GOAL_EX.png", transitio[0][1])
                    plt.close()
                #"d4[d2d1]Ed3"

                #"[d4d3][d2d1]EE"
                all_the_unique_actions.append(str([part1_trans_, part2_trans_]))


    print("LEN all_the_unique_actions: {}".format(len(all_the_unique_actions))) # 1452

    if add_noisy_trans:

        random.shuffle(all_the_unique_actions)


        print("int(100/perc)")
        print(100/perc)

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

        print("all_the_unique_actions")
        print(len(all_the_unique_actions))
        print(len(all_the_unique_actions_noisy_part))

  
        # 367
        # 1452 798 143
        #  1452-798 + 798-143 + 143*3
        # = 654 + 655 + 429
        # = 1738 !!!

        # case where 1% of actions have separated labels
        #       
        #        ==> 14 
        #
        #      1452 - 798  +   798 - 14  +    14*3  ==== 1480 !!!!
        #           



        # check if in all_the_unique_actions_ten_percent_noisy_and_dup there is at least one
        # which belong to longest_plan_transis
        one_belong = False
        for trrrr in all_the_unique_actions_ten_percent_noisy_and_dup:
            if trrrr in longest_plan_transis:
                print("at least one noisy and differentiated action belong to the longest plan")



        all_the_unique_actions_noisy_part_count = {key: 0 for key in all_the_unique_actions_noisy_part}

        print("Number of UNIQUE noisy transs {}".format(str(len(all_the_unique_actions_noisy_part_count)))) # 81


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

    unique_obs_img_preproc, orig_max, orig_min = preprocess(reduced_uniques, histo_bins=24)
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)
    
    longest_plan_transis_test = []

    print(len(all_the_unique_actions_noisy_part_count))
    print(len(all_the_unique_actions_ten_percent_noisy_and_dup))
    print("exp_folder_")
    print(exp_folder_)



    test_noisy_transis = []

    already_created = False
    # main loop 
    for iii, trace in enumerate(loaded_traces):

        nb_additional_noisy_trans = 0

        actions_for_one_trace = []

        for trtrtr, transitio in enumerate(trace):



            if remove_a_trans:

                first_split_ = transitio[1][3].split("}, {")
                part1_trans_ = first_split_[0].replace('[{', '')
                part2_trans_ = first_split_[1].replace('}]', '')

                if replace_in_str(part1_trans_) in trans_to_remove_list and replace_in_str(part2_trans_) in trans_to_remove_list:
                    
                    continue


            # normalize the two images
            transi_0_prepro, _, _ = preprocess(reduce_resolution(transitio[0][0]), histo_bins=24)
            transi_0_reduced_and_norm, _, _ = normalize_colors(transi_0_prepro, mean=mean_all, std=std_all)

            transi_1_prepro, _, _ = preprocess(reduce_resolution(transitio[0][1]), histo_bins=24)
            transi_1_reduced_and_norm, _, _ = normalize_colors(transi_1_prepro, mean=mean_all, std=std_all)
            
            ## each "friendly" version of the actions' states is created here
            first_split_ = transitio[1][3].split("}, {")
            part1_trans_ = replace_in_str(first_split_[0].replace('[{', ''))
            part2_trans_ = replace_in_str(first_split_[1].replace('}]', ''))

            already_created = False





            if trtrtr % 50 == 0:
                save_np_image(exp_folder, transi_0_reduced_and_norm, "pair_"+str(trtrtr)+"_0")
                save_np_image(exp_folder, transi_1_reduced_and_norm, "pair_"+str(trtrtr)+"_1")




            #### SAVING THE INIT/GOAL IMAGES (non noisy version)
            if create_init_goal and not already_created:
                already_created = True
                if part1_trans_ == init_st.replace("_noise", ""):
                    print("INIT SAVED")
                    save_np_image(exp_folder_, transi_0_reduced_and_norm, "init_NoNoise")
                    # plt.imsave("HANOI_INIT-val.png", reduce_resolution(transitio[0][0]))
                    # plt.close()
                if part2_trans_ == init_st.replace("_noise", ""):
                    print("INIT SAVED")
                    save_np_image(exp_folder_, transi_1_reduced_and_norm, "init_NoNoise")
                    # plt.imsave("HANOI_INIT-val.png", reduce_resolution(transitio[0][1]))
                    # plt.close()

                if part1_trans_ == goal_st.replace("_noise", ""):
                    print("GOAL SAVED")
                    save_np_image(exp_folder_, transi_0_reduced_and_norm, "goal_NoNoise")
                
                    # im2 = unnormalize_colors(transi_0_reduced_and_norm, mean_all, std_all)
                    # im2 = deenhance(im2)
                    # im2 = denormalize(im2, orig_min, orig_max)
                    # im2 = np.clip(im2, 0, 1)


                    # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(transitio[0][0]))
                    # plt.close()
                if part2_trans_ == goal_st.replace("_noise", ""):
                    print("GOAL SAVED")
                    save_np_image(exp_folder_, transi_1_reduced_and_norm, "goal_NoNoise")

                    # im2 = unnormalize_colors(transi_1_reduced_and_norm, mean_all, std_all)
                    # im2 = deenhance(im2)
                    # im2 = denormalize(im2, orig_min, orig_max)
                    # im2 = np.clip(im2, 0, 1)

                    # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(transitio[0][1]))
                    # plt.close()



            # If the current transition is not NOISY at all
            if str([part1_trans_, part2_trans_]) not in all_the_unique_actions_noisy_part_count:

                all_images_reduced_and_norm.append(transi_0_reduced_and_norm)
                all_images_reduced_and_norm.append(transi_1_reduced_and_norm)
                
                actions_for_one_trace.append(str([part1_trans_, part2_trans_]))

                # ADD the action to the unique vectors
                if str([part1_trans_, part2_trans_]) not in all_actions_unique:
                    all_actions_unique.append(str([part1_trans_, part2_trans_]))
                    all_transitions_unique.append([part1_trans_, part2_trans_])

                # ADDING THE IMAGES to a unique set
                if str(part1_trans_) not in all_images_reduced_and_norm_uniques_nonnoisy:
                    all_images_reduced_and_norm_uniques_nonnoisy[str(part1_trans_)] = transi_0_reduced_and_norm

                if str(part2_trans_) not in all_images_reduced_and_norm_uniques_nonnoisy:
                    all_images_reduced_and_norm_uniques_nonnoisy[str(part1_trans_)] = transi_1_reduced_and_norm

            # if the trans is NOISY
            elif add_noisy_trans and str([part1_trans_, part2_trans_]) in all_the_unique_actions_noisy_part_count:

                # update the count
                all_the_unique_actions_noisy_part_count[str([part1_trans_, part2_trans_])] += 1
                
                # produce number_noisy_versions versions of the transitons

                for inndex in range(number_noisy_versions):

                    nb_additional_noisy_trans += 1

                    # make the noisy colored normalized array of both states
                    if 'hsv' in noise_type:
                        
                        noisy1 = rgb_to_hsv_with_pil(transitio[0][0])
                        noisy1 = add_gaussian_noise_to_hsv(noisy1, mean, std, noise_type.split("_")[1])
                        noisy1 = np.array(noisy1.convert(mode="RGB"))
                        
                        noisy2 = rgb_to_hsv_with_pil(transitio[0][0])
                        noisy2 = add_gaussian_noise_to_hsv(noisy2, mean, std, noise_type.split("_")[1])
                        noisy2 = np.array(noisy2.convert(mode="RGB"))
                        
                        noisy1_preproc, _, _ = preprocess(reduce_resolution(noisy1), histo_bins=24)
                        noisy1_reduced_preproc_norm, _, _ = normalize_colors(noisy1_preproc, mean=mean_all, std=std_all)

                        noisy2_preproc, _, _ = preprocess(reduce_resolution(noisy2), histo_bins=24)
                        noisy2_reduced_preproc_norm, _, _ = normalize_colors(noisy2_preproc, mean=mean_all, std=std_all)


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


                        # immmm = transitio[0][1]
                        # np.random.seed(55)
                        # gaussian_noise_1 = np.random.normal(mean, 20, immmm.shape)
                        # immmm = immmm + gaussian_noise_1
                        # immmm = np.clip(immmm, 0, 255).astype(np.uint8) 
                        # plt.imsave("HANOI_NOISY2.png", immmm)
                        # plt.close()
                        # exit()



                        # unorm = unnormalize_colors(noisy1, mean_all, std_all) 
                        # dehanced = deenhance(unorm)
                        # denormalized = denormalize(dehanced, orig_min, orig_max)
                        # denormalized = np.clip(denormalized, 0, 1)
                        # plt.imsave("HANOI_NOISY1.png", denormalized)
                        # plt.close()

                    if str(part1_trans_) not in all_images_reduced_and_norm_uniques_noisy:
                        all_images_reduced_and_norm_uniques_noisy[str(part1_trans_)] = noisy1_reduced_preproc_norm

                    if str(part2_trans_) not in all_images_reduced_and_norm_uniques_noisy:
                        all_images_reduced_and_norm_uniques_noisy[str(part1_trans_)] = noisy2_reduced_preproc_norm


                    # if 'noise' init/state was chosen
                    # AND if one of the present states is init/state
                    # save the colored normalized array in some picke file
                    if create_init_goal:

                        #if 'noise' in init_st:
                        if part1_trans_ == init_st.replace("_noise", ""):
                            save_np_image(exp_folder_noisy, noisy1, "init-noise-"+str(inndex))

                        if part2_trans_ == init_st.replace("_noise", ""):
                            save_np_image(exp_folder_noisy, noisy2, "init-noise-"+str(inndex))

                        #if 'noise' in goal_st:

                        if part1_trans_ == goal_st.replace("_noise", ""):
                            # print("laaaaa")
                            # exit()
                            save_np_image(exp_folder_noisy, noisy1, "goal-noise-"+str(inndex))
                            # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(noisy1))
                            # plt.close()

                        if part2_trans_ == goal_st.replace("_noise", ""):
                            # print("laaaaa1111111111")
                            # exit()
                            save_np_image(exp_folder_noisy, noisy2, "goal-noise-"+str(inndex))
                            # plt.imsave("HANOI_GOAL-val.png", reduce_resolution(noisy2))
                            # plt.close()


                    # fill in the all_images_reduced_and_norm with the noisy images
                    all_images_reduced_and_norm.append(noisy1_reduced_preproc_norm)
                    all_images_reduced_and_norm.append(noisy2_reduced_preproc_norm)

                    # If the trans labels must be differentiated !
                    if str([part1_trans_, part2_trans_]) in all_the_unique_actions_ten_percent_noisy_and_dup:
                        actions_for_one_trace.append(str([part1_trans_, part2_trans_])+"_"+str(inndex))

                        # ADD the action to the unique vectors
                        if str([part1_trans_, part2_trans_])+"_"+str(inndex) not in all_actions_unique:
                            all_actions_unique.append(str([part1_trans_, part2_trans_])+"_"+str(inndex))
                            all_transitions_unique.append([part1_trans_+"_"+str(inndex), part2_trans_+"_"+str(inndex)])
                    # or notall_transitions_unique

                    else:
                        actions_for_one_trace.append(str([part1_trans_, part2_trans_]))

                # ADD the action to the unique vectors
                if str([part1_trans_, part2_trans_]) not in all_actions_unique and str([part1_trans_, part2_trans_]) not in all_the_unique_actions_ten_percent_noisy_and_dup:
                    all_actions_unique.append(str([part1_trans_, part2_trans_]))
                    all_transitions_unique.append([part1_trans_, part2_trans_])

        print("LEN actions_for_one_trace {}".format(str(len(actions_for_one_trace))))

        # add the actions of this trace to all_actions_for_trace (the array of all actions of all traces)
        all_actions_for_trace.append(actions_for_one_trace)

        # add the indices for this trace to traces_indices array
        traces_indices.append([start_trace_index, start_trace_index+(len(trace)+nb_additional_noisy_trans)*2])
        
        # update the index for the next trace
        start_trace_index+=(len(trace)+nb_additional_noisy_trans)*2


    # print(longest_plan_transis_test)
    # 
    print("len all_actions_unique")
    print(len(all_actions_unique))
    # print("len(all_images_reduced)") # 3194
    # print(len(all_images_reduced))
    # print(test_noisy_transis)

    


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

    print("laaaaaa")
    with open("AAAAll_actionsPRE.txt","w") as f:
    
        for i in range(len(all_actions_unique)):
            #np.argmax(all_actions_one_hot[i]
            f.write("a"+str(i)+" is "+str(all_actions_unique[i])+"\n")


    # si trans parcouru == "[{"+part2_trans+"}, {"+part1_trans+"}]"



    # all_images_reduced > gaussian > clip > preprocess > normalize_colors

   
    # counntteerr=0
    # all_images_reduced_gaussian_20 = add_noise(all_images_reduced, seed=1)
    # counntteerr=0
    # all_images_reduced_gaussian_30 = add_noise(all_images_reduced, seed=2)
    # counntteerr=0
    # all_images_reduced_gaussian_40 = add_noise(all_images_reduced, seed=3)

    # save_noisy("hanoi_4_4_dataset", "all_images_seed1.p", all_images_reduced_gaussian_20)
    # save_noisy("hanoi_4_4_dataset", "all_images_seed2.p", all_images_reduced_gaussian_30)
    # save_noisy("hanoi_4_4_dataset", "all_images_seed3.p", all_images_reduced_gaussian_40)


    # loaded_noisy1 = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/all_images_seed1.p")
    # all_images_seed1 = loaded_noisy1["images"]

    # loaded_noisy2 = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/all_images_seed2.p")
    # all_images_seed2 = loaded_noisy2["images"]

    # loaded_noisy3 = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/all_images_seed3.p")
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


    print("len all_actions_unique")



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

    print("train_set all_actions_unique")
    print(len(train_set[0][1]))


    return train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min




#train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()


# print('len all_actions_unique')
# print(len(all_actions_unique))
# exit()

# print("all_actions_unique {}".format(len(all_actions_unique))) # 933

# print("all_actions_one_hot {}".format(len(all_actions_one_hot))) # 2799


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


#     plt.imsave("hanoi_4-4-pair_"+str(hh)+"_pre.png", im1)
#     plt.close()

#     plt.imsave("hanoi_4-4-pair_"+str(hh)+"_suc.png", im2)
#     plt.close()


# exit()


