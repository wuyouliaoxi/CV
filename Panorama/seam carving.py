import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter1d

"""
img1: common-kestrel.jpg
img2: kingfishers.jpg using removing with mask
img3: kingfishers.jpg using seam carving with mask
img4: downscaling vincent-on-cliff.jpg
img5: upscaling vincent-on-cliff.jpg

"""

# https://zhuanlan.zhihu.com/p/38974520
# Exercise 2.4: Removing columns and rows
def energy_function(img):
    con_0 = np.abs(signal.convolve2d(img[:, :, 0], laplacian, mode='same'))
    con_1 = np.abs(signal.convolve2d(img[:, :, 1], laplacian, mode='same'))
    con_2 = np.abs(signal.convolve2d(img[:, :, 2], laplacian, mode='same'))
    return con_0 + con_1 + con_2


def removing(img):
    for _ in range(10):
        cost = energy_function(img)
        row_cost = np.sum(cost, axis=1)
        row_idx = np.argsort(row_cost)
        col_cost = np.sum(cost, axis=0)
        col_idx = np.argsort(col_cost)
        img = np.delete(img, row_idx[0:100], 0)
        img = np.delete(img, col_idx[0:150], 1)
    return img


def mask_remove(img, mask):
    for _ in range(20):
        cost = np.add(energy_function(img), mask)  # add mask to the cost array
        row_cost = np.sum(cost, axis=1)  # sum vertically
        row_idx = np.argsort(row_cost)
        col_cost = np.sum(cost, axis=0)  # sum horizontally
        col_idx = np.argsort(col_cost)
        img = np.delete(img, row_idx[0:10], 0)
        img = np.delete(img, col_idx[0:20], 1)
        mask = np.delete(mask, row_idx[0:10], 0)
        mask = np.delete(mask, col_idx[0:20], 1)
    return img


# Exercise 2.5: path Finding
# cumulative cost
def cum_cost(cost):
    cum = np.ones_like(cost)  # same shape as cost
    cum[0, :] = cost[0, :]  # fix the first row in cost
    for i in range(1, cum.shape[0]):  # val(x,y) = min(val((x-1:x+2),y-1)+cost(x,y)
        cum[i, :] = cost[i, :] + minimum_filter1d(cum[i - 1, :], size=3, mode='reflect')  # row-wise
    return cum


def path(img):
    rows, cols = img.shape
    idx_whole = np.zeros(rows)
    idx_cols = np.zeros(rows)
    index = np.argmin(img[-1, :])  # index of column in a row
    idx_whole[-1] = (rows - 1) * cols + index  # write the index in whole the 2d-array
    idx_cols[-1] = index
    for i in reversed(range(0, rows - 1)):
        if index == 0:  # in case for the left-most column
            index = np.argmin(img[i, index: index + 2])
        elif index == cols - 1:  # in case for the right-most column
            index = index - (1 - np.argmin(img[i, index - 1: index + 1]))
        else:
            index = index - (1 - np.argmin(img[i, index - 1: index + 2]))
        idx_whole[i] = i * cols + index
        idx_cols[i] = index

    left, right = path_neighbor(idx_cols, cols)
    return idx_whole, left, right


# Exercise 2.6: use seam carving
def remove_path(img, path):
    global new_img
    if len(img.shape) == 2:  # in case img is 2D-mask
        rows, cols = img.shape
        img = img.reshape(rows * cols)  # first convert to 1D-array
        new_img = np.delete(img, path.astype(int), axis=0).reshape(rows, cols - 1)
    if len(img.shape) == 3:
        rows, cols, rbg = img.shape
        img = img.reshape(rows * cols, rbg)
        new_img = np.delete(img, path.astype(int), axis=0).reshape(rows, cols - 1, rbg)
    return new_img


# Exercise 2.7: seam carving on images
# make it faster, find the neighborhood of the path
def path_neighbor(idx_cols, cols):
    path_mean = np.mean(idx_cols)  # the approximate location of the path
    path_std = np.std(idx_cols)  # Compute the standard deviation of the path
    left = np.clip(np.round(path_mean - path_std).astype(int), 0, cols) # left bound of the path
    right = np.clip(np.round(path_mean + path_std + 1).astype(int), 0, cols) # right bound of the path
    return left, right

def seam_carving(img, cost, mask=None):
    cum = cum_cost(cost)
    path_idx, left, right = path(cum)
    new_img = remove_path(img, path_idx)  # remove path
    new_cost = remove_path(cost, path_idx)  # update the cost
    new_cost[:, left:right] = energy_function(new_img[:, left:right])  # update only the cost of neighbor
    new_mask = mask
    if mask is not None:
        new_mask = remove_path(mask, path_idx)
        new_cost += new_mask
    return new_img, new_cost, new_mask


# Exercise 2.8: upscaling images
def insert_path(img, path_idx):
    global new_img
    path_idx = path_idx.astype(int)
    if len(img.shape) == 2:
        rows, cols = img.shape  # mask is 2D
        img = img.reshape(rows * cols)
        path = img[path_idx]
        r = img[path_idx + 1]
        new_path = np.round((path + r) / 2) # use the average of the values to fill the gap
        new_img = np.insert(img, path_idx + 1, new_path, axis=0).reshape(rows, cols + 1)
    if len(img.shape) == 3:
        rows, cols, rbg = img.shape
        img = img.reshape(rows * cols, rbg)
        path = img[path_idx, :]
        r = img[path_idx + 1, :]
        new_path = np.round((path.astype(int) + r.astype(int)) / 2)
        new_img = np.insert(img, path_idx + 1, new_path, axis=0).reshape(rows, cols + 1, rbg)
    return new_img


def insert_path_mask(mask, path_idx):
    path_idx = path_idx.astype(int)
    rows, cols = mask.shape
    mask = mask.reshape(rows * cols)
    new_path = 255 * np.ones_like(path_idx)  # create a white new path in mask
    new_mask = np.insert(mask, path_idx + 1, new_path, axis=0)  # insert a column
    new_mask[path_idx -1] += new_path  # insert 255 on the left of path
    new_mask[path_idx] += new_path
    new_mask = new_mask.reshape(rows, cols + 1)
    return new_mask


def upscaling(img, cost, mask):
    cum = cum_cost(cost)
    path_idx, left, right = path(cum)
    new_img = insert_path(img, path_idx)
    new_mask = insert_path_mask(mask, path_idx)  # update mask
    new_cost = energy_function(new_img) + new_mask  # update cost
    return new_img, new_cost, new_mask


# upscaling or downscaling
def up_or_down(img, mask=None, mode='downscaling'):
    rows, cols, _ = img.shape
    crop_iter_num = int(CROP_SCALE * rows)
    extend_iter_num = int(EXTEND_SCALE * rows)

    if mask is not None:
        cost = energy_function(img) + mask
    else:
        cost = energy_function(img)

    # seam_mask = np.zeros_like(cost)

    if mode == 'downscaling':
        for _ in range(crop_iter_num):
            img, cost, mask = seam_carving(img, cost, mask)
    if mode == 'upscaling':
        for _ in range(extend_iter_num):
            img, cost, mask = upscaling(img, cost, mask)
    # plt.imshow(mask)
    # plt.show()

    return img


def better_mask(mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] < 80:
                mask[i][j] = 0
            else:
                mask[i][j] = 255
    return mask


laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
CROP_SCALE = 0.5  # downscaling to 50%
EXTEND_SCALE = 0.3  # upscaling to 130%


if __name__ == "__main__":
    while True:
        show_result = input('which image do u want to show? ')
        show_result = int(show_result)
        if show_result == 1:
            # img1: common-kestrel.jpg
            print('img1: common-kestrel.jpg')
            img_kestrel = np.array(Image.open('common-kestrel.jpg'))  # (1333, 2000, 3)
            crop_kingfishers = removing(img_kestrel)
            plt.subplot(121)
            plt.imshow(img_kestrel)
            plt.title('common-kestrel')
            plt.subplot(122)
            plt.imshow(crop_kingfishers)
            plt.title('remove rows and columns')
            plt.show()
        if show_result == 2:
            # img2: kingfishers.jpg
            print('img2: kingfishers.jpg using removing with mask')
            img_kingfishers = np.array(Image.open('kingfishers.jpg'))
            protect_mask = np.array(Image.open('kingfishers-mask.png').convert('L'))  # convert to grayscale
            crop_kingfishers = mask_remove(img_kingfishers, protect_mask)
            plt.subplot(121)
            plt.imshow(img_kingfishers)
            plt.title('kingfishers')
            plt.subplot(122)
            plt.imshow(crop_kingfishers)
            plt.title('mask_remove')
            plt.show()
        if show_result == 3:
            print('img3: kingfishers.jpg using seam carving with mask')
            print('wait 1 minute')
            img_kingfishers = np.array(Image.open('kingfishers.jpg'))
            protect_mask = np.array(Image.open('kingfishers-mask.png').convert('L'))  # convert to grayscale
            result_down = up_or_down(img_kingfishers, protect_mask, mode='downscaling')
            plt.subplot(121)
            plt.imshow(img_kingfishers)
            plt.title('kingfishers')
            plt.subplot(122)
            plt.imshow(result_down)
            plt.title('crop')
            plt.show()
        if show_result == 4:
            # img3: vincent-on-cliff.jpg
            print('img4: downscaling vincent-on-cliff.jpg')
            print('wait 1 Minute')
            img_vincent = np.array(Image.open('vincent-on-cliff.jpg'))
            protect_mask = np.array(Image.open('vincent-on-cliff-mask.jpg').convert('L'))
            protect_mask = better_mask(protect_mask)
            result_down = up_or_down(img_vincent, protect_mask, mode='downscaling')
            plt.subplot(121)
            plt.imshow(img_vincent)
            plt.title('vincent')
            plt.subplot(122)
            plt.imshow(result_down)
            plt.title('crop')
            plt.show()
        if show_result == 5:
            print('img5: upscaling vincent-on-cliff.jpg')
            print('wait 5 Minute')
            img_vincent = np.array(Image.open('vincent-on-cliff.jpg'))
            protect_mask = np.array(Image.open('vincent-on-cliff-mask.jpg').convert('L')).astype(int)
            # protect_mask = None
            protect_mask = better_mask(protect_mask)
            result_up = up_or_down(img_vincent, protect_mask, mode='upscaling')
            plt.subplot(121)
            plt.imshow(img_vincent)
            plt.title('vincent')
            plt.subplot(122)
            plt.imshow(result_up)
            plt.title('extended')
            plt.show()
