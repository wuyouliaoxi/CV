import matplotlib.pyplot as plt
import numpy
from scipy import signal
from scipy.ndimage import minimum_filter1d
from scipy.signal import convolve
import numpy as np
from PIL import Image
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
def energy_function(img):
    con_0 = np.abs(signal.convolve2d(img[:, :, 0], laplacian, mode='same'))
    con_1 = np.abs(signal.convolve2d(img[:, :, 1], laplacian, mode='same'))
    con_2 = np.abs(signal.convolve2d(img[:, :, 2], laplacian, mode='same'))
    return con_0 + con_1 + con_2


def cum_cost(cost):
    cum = np.ones_like(cost)  # same shape as cost
    cum[0, :] = cost[0, :]  # fix the first row in cost
    for i in range(1, cum.shape[0]):  # val(x,y) = min(val((x-1:x+2),y-1)+cost(x,y)
        cum[i, :] = cost[i, :] + minimum_filter1d(cum[i - 1, :], size=3, mode='reflect')  # row-wise
    return cum


def path(arr):
    rows, cols = arr.shape
    path_idx = np.zeros(rows)
    path_cols = np.zeros(rows)
    j = np.argmin(arr[-1, :])  # j is index of column in a row
    path_idx[-1] = (rows - 1) * cols + j  # write the index in whole the 2d-array
    path_cols[-1] = j
    for i in reversed(range(0, rows - 1)):
        if j == 0:
            j = np.argmin(arr[i, 0: 2])
        elif j == cols - 1:
            j = j - (1 - np.argmin(arr[i, j - 1: j + 1]))
        else:
            j = j - (1 - np.argmin(arr[i, j - 1: j + 2]))
        path_idx[i] = i * cols + j
        path_cols[i] = j

    return path_idx, path_cols


arr = np.array(Image.open('vincent-on-cliff.jpg'))
arr = energy_function(arr)
arr1 = cum_cost(arr)

path_idx, path_cols = path(arr1)



def remove_path(arr, path_idx):
    global new_arr
    if len(arr.shape) == 2:
        rows, cols = arr.shape
        arr = arr.reshape(rows * cols)  # first convert to 1D-array
        new_arr = np.delete(arr, path_idx.astype(int), axis=0).reshape(rows, cols - 1)
    if len(arr.shape) == 3:
        rows, cols, deps = arr.shape
        arr = arr.reshape(rows * cols, deps)
        new_arr = np.delete(arr, path_idx.astype(int), axis=0).reshape(rows, cols - 1, deps)
    return new_arr





def insert_path_mask(mask, path_idx):
    path_idx = path_idx.astype(int)
    mask = mask.astype(int)
    rows, cols = mask.shape
    mask = mask.reshape(rows * cols)
    new_path = 255 * np.ones_like(path_idx)  # create a white new path in mask
    print(new_path.shape)
    new_mask = np.insert(mask, path_idx + 1, new_path, axis=0)
    print(new_mask[path_idx])
    for i in range(-1, 1):
        new_mask[path_idx + i] += new_path
    new_mask = new_mask.reshape(rows, cols + 1)
    return new_mask


mask = np.array(Image.open('vincent-on-cliff-mask.jpg').convert('L'))
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i][j]<80:
            mask[i][j]=0
        else:
            mask[i][j]= 255

# print('mask: ', mask)
# plt.subplot(111)


a = insert_path_mask(mask, path_idx)
print(a.shape)
plt.imshow(a)
plt.show()
