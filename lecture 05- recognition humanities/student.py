'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation as segmentation
import numpy as np
from skimage.color import rgb2hsv


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    segement_mask = segmentation.felzenszwalb(im_orig, scale, sigma, min_size)
    seg_img = np.zeros((im_orig.shape[0], im_orig.shape[1], 4))
    seg_img[:, :, :3] = im_orig
    seg_img[:, :, 3] = segement_mask
    return seg_img


def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    sum = 0
    for a, b in zip(r1["hist_color"], r2["hist_color"]):
        con = np.concatenate(([a], [b]), axis=0)
        sum += np.sum(np.min(con, axis=0))
    return sum


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    sum = 0
    for a, b in zip(r1["hist_text"], r2["hist_text"]):
        con = np.concatenate(([a], [b]), axis=0)
        sum += np.sum(np.min(con, axis=0))
    return sum


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    return 1 - (r1["size"] + r2["size"]) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    max_x = max(r1["max_x"], r2["max_x"])
    min_x = min(r1["min_x"], r2["min_x"])
    max_y = max(r1["max_y"], r2["max_y"])
    min_y = min(r1["min_y"], r2["min_y"])
    bb_size = (max_x - min_x) * (max_y - min_y)
    return 1 - (bb_size - r1["size"] - r2["size"]) / imsize


def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))


def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = []
    ### YOUR CODE HERE ###
    for i in range(3):
        h, _ = np.histogram(img[:, i], bins=BINS)
        hist.append(h)
    #  normalization
    return np.array(hist) / len(img)


def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    # LBP is an invariant descriptor that can be used for texture classification.
    for i in range(3):
        ret[:, :, i] = skimage.feature.local_binary_pattern(img[:, :, i], P=8, R=1)
    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = []
    ### YOUR CODE HERE ###
    for i in range(3):
        h, _ = np.histogram(img[:, i], bins=BINS)
        hist.append(h)
    # L1 normalize
    return np.array(hist) / len(img)


def extract_regions(img):
    """
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    """
    R = {}
    ### YOUR CODE HERE ###
    # Convert image to hsv color map
    hsv_img = rgb2hsv(img[:, :, :3])
    mask = img[:, :, 3]
    texture_gradient = calc_texture_gradient(img[:, :, :-1])
    for i in range(int(np.max(mask))):
        y_set, x_set = np.where(mask == i)
        masked_area = mask == i
        R[i] = {
            "labels": i,
            "size": np.sum(masked_area == 1),
            "min_x": np.min(x_set),
            "max_x": np.max(x_set),
            "min_y": np.min(y_set),
            "max_y": np.max(y_set),
            "hist_color": calc_colour_hist(hsv_img[masked_area]),
            "hist_text": calc_texture_hist(texture_gradient[masked_area])
        }
    return R


def extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    for i in range(len(regions)):
        for j in range(len(regions)):
            if i in regions.keys() and j in regions.keys() and i < j:
                if intersect(regions[i], regions[j]):
                    n1 = (i, regions[i])
                    n2 = (j, regions[j])
                    neighbours.append((n1, n2))
    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    ### YOUR CODE HERE
    rt = {
        "labels": r1["labels"] + r2["labels"],
        "size": new_size,
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "hist_color": (r1["hist_color"] * r1["size"] + r2["hist_color"] * r2["size"]) / new_size,
        "hist_text": (r1["hist_text"] * r1["size"] + r2["hist_text"] * r2["size"]) / new_size
    }
    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    # ai = key = label, ar = value = similarity
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get largest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        reg2remove = []
        for key in S.keys():
            if (i in key) or (j in key):
                reg2remove.append(key)

        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for k in reg2remove:
            del S[k]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        for (x, y) in reg2remove:
            if (x, y) != (i, j):
                if x != i:
                    new = x
                elif y != j:
                    new = y
                S[(t, new)] = calc_sim(R[t], R[new], imsize)

    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for r in R.values():
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
    return image, regions
