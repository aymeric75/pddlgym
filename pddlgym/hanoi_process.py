import numpy as np
from skimage import exposure, color


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
    unnormalized_images = normalized_images * (std + 1e-6) + mean
    return np.round(unnormalized_images).astype(np.uint8)

def normalize_colors(images, mean=None, std=None, second=False):    

    if mean is None or std is None:

        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)

    return (images - mean)/(std+1e-20), mean, std

