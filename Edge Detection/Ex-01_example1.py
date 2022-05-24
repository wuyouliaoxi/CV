import random
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import medfilt

# show amplitude & distance image
def show_image(matrix, title):
    plt.imshow(matrix)
    plt.title(title)
    plt.show()

# invert point cloud into a matrix with only 3 columns
def reshape_pc(matrix):
    matrix = matrix.reshape(len(matrix) * len(matrix[0]), 3)
    return matrix

# show 3D point cloud
def show_point_cloud(matrix, subsample, title):
    ax = plt.subplot(111, projection='3d')
    x = matrix[::subsample, 0]
    y = matrix[::subsample, 1]
    z =  matrix[::subsample, 2]
    ax.scatter(x, y, z)
    ax.set_title(title)
    plt.show()

# using 3 points to determine a plane
def find_plane(point1, point2, point3):
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = point2 - point1
    AC = point3 - point1
    # to get the normal vector of the plane
    n = np.cross(AB, AC)
    # calculate d with nx=d
    d = np.dot(n,point1)
    return n, d

# calculate the distance (a set of points) from another point to the plane
def cal_distance(point1, point2, point3, point4):
    n, d = find_plane(point1, point2, point3)
    numerator = abs(n[0] * point4[:, 0] + n[1] * point4[:, 1] + n[2] * point4[:, 2] - d)
    denominator = np.sqrt(np.sum(np.square([n])))
    distance = numerator / denominator
    return distance

# use RANSAC algorithm to find the plane models with the most inliers
def ransac(matrix, threshold, N):
    max_inlier = 0
    best_plane = None
    for i in range(N):
        list = range(len(matrix))
        samples = random.sample(list, 3)
        point1 = matrix[samples[0]]
        point2 = matrix[samples[1]]
        point3 = matrix[samples[2]]
        dis_array = cal_distance(point1, point2, point3, matrix)
        # invert int to bool, note that true as 1, false as 0
        model = dis_array < threshold # shape should be (len(matrix),) and type is bool
        n_inliers = np.count_nonzero(model)
        if n_inliers > max_inlier:
            max_inlier = n_inliers
            best_plane = model
    return max_inlier, best_plane

# because the best_plane is bool
# convert single column array to a 3-columns array
def mask_image(matrix, model):
    model1 = np.c_[model, model, model] # model is a single column array, so convert to 3 column
    new_model = matrix[model1] # shape is changed to one row array
    new_model = np.reshape(new_model, (len(new_model) // 3, 3)) # reshape to 3 column
    return new_model

# inliers = 1, outliers = 0, but dilation to the bigger shape
def visualize_mask(shape_bool, plane_model): # they should be both single column
    mask_result = np.zeros(len(shape_bool), dtype=bool)
    j = 0
    for i in range(len(shape_bool)):
        if shape_bool[i] == True:
            mask_result[i] = plane_model[j]
            j = j + 1
        else:
            mask_result[i] = shape_bool[i]
    return mask_result

# morphology.binary_closing operator
def closing_mask(matrix, iteration):
    result = scipy.ndimage.morphology.binary_closing(matrix, iterations = iteration)
    return plt.imshow(result)

# morphology.binary_opening operator
def opening_mask(matrix, iteration):
    result = scipy.ndimage.morphology.binary_opening(matrix, iterations = iteration)
    return plt.imshow(result)

# opening operator without show
def opening_mask_noShow(matrix, iteration):
    result = scipy.ndimage.morphology.binary_opening(matrix, iterations = iteration)
    return result


# closing operator without show
def closing_mask_noShow(matrix, iteration):
    result = scipy.ndimage.morphology.binary_closing(matrix, iterations = iteration)
    return result

def label_opening(matrix, iteration):
    matrix2 = opening_mask_noShow(matrix, iteration)
    labeled_array, num_features = scipy.ndimage.label(matrix2)
    print('number of features with opening Operators: ', num_features)
    return opening_mask(matrix, iteration)

def label_closing(matrix, iteration):
    matrix2 = closing_mask_noShow(matrix, iteration)
    labeled_array, num_features = scipy.ndimage.label(matrix2)
    print('number of features with closing Operators: ', num_features)
    return closing_mask(matrix, iteration)

# calculate the height of the box
def cal_height2(top_plane_3D, floor_plane_3D, iteration):
    height = 0
    for i in range(iteration):
        list = range(len(floor_plane_3D))
        samples = random.sample(list, 3)
        point1_floor = floor_plane_3D[samples[0]]
        point2_floor = floor_plane_3D[samples[1]]
        point3_floor = floor_plane_3D[samples[2]]
        height = height + np.mean(cal_distance(point1_floor, point2_floor, point3_floor, top_plane_3D))
    height_average = height / iteration
    return height_average

# calculate all the distance of from every two points
def smallst_dis(number, list1):
    listNew = []
    j = 1
    while number + j <= 5:
        distance = np.sqrt(np.sum(np.square(list1[number] - list1[number + j])))
        listNew.append(distance)
        j = j + 1
    return min(listNew)


def smallestDis(point_, point_list):
    listDis = []
    for i in range(len(point_list)):
        distance = np.sum(np.square(point_list[i] - point_))
        listDis.append(distance)
    return min(listDis)

# find the corresponded smallstIndex in corners1 and corners2
def returnProjection(smallstIndex_inpt, corners1_input, corners2_input):
    if smallstIndex_inpt <= 2:
        return corners1_input[smallstIndex_inpt]
    else:
        return corners2_input[smallstIndex_inpt - 3]

# ----------------------------------
# Step1. Getting and reading the data
# load the examples from matlab file
mat1=scipy.io.loadmat("example1kinect.mat")

# show the amplitude image
show_image(mat1['amplitudes1'], 'amplitude image')

# show the distance image
show_image(mat1['distances1'], 'distance image')

# point cloud visualization
cloud1 = mat1['cloud1'] # shape: (424, 512, 3)
cloud1 = reshape_pc(cloud1) # shape: (217088, 3: X,Y,Z)
show_point_cloud(cloud1, 8, 'point cloud with subsampling 8')

# -------------------------------------
# Step2. RANSAC
# ignoring the points which the z-component of a vector is 0, that means all points are valid
filted_cloud1 = cloud1[:, 2] > 0  # bool shape：(217088,)
cloud1_nonzeroz = mask_image(cloud1, filted_cloud1) # shape: (202007, 3), remove all of the false points

# ransac to find floor plane model and output the inliers
floor_inlier, floor_plane = ransac(cloud1_nonzeroz, 0.05, 500) # shape of floor plane: (202007, )
print('max inliers of floor plane:', floor_inlier)

# because the obtained best floor plane is bool, it must be converted
# The points that belong to the floor plane are labeled True, and delete the points that not belong to floor
# remove the points that not belong to the floor plane
floor_plane_3D = mask_image(cloud1_nonzeroz, floor_plane) # shape: (~150000,3)
show_point_cloud(floor_plane_3D, 4, '3D floor plane with subsample 4')

# now we want to show the distance image of the floor plane
# mask image that inliers of the floor=1 and other as outliers=0
# shape of bool filted_cloud1 : (217088,),
mask2 = visualize_mask(filted_cloud1, floor_plane) # fill the image to the original shape
floor_plane_2D = np.reshape(mask2, mat1['distances1'].shape) # shape: (424,512)
show_image(floor_plane_2D, '2D floor plane')

# --------------------------------
# Step3. Filtering on the Mask Image
# morphological operator: opening
# Result: The iteration times can not be too big, otherwise it will make noisier
# opening operator: remove the small regions
plt.figure()
plt.suptitle('opening operator with different iterations')
plt.subplot(221)
opening_mask(floor_plane_2D, 1)
plt.title('iteration=1')
plt.subplot(222)
opening_mask(floor_plane_2D, 2)
plt.title('iteration=2')
plt.subplot(223)
opening_mask(floor_plane_2D, 3)
plt.title('iteration=3')
plt.subplot(224)
opening_mask(floor_plane_2D, 4)
plt.title('iteration=4')
plt.show()


# morphological operator: closing
# closing: remove the holes
# in case: time=500, dis=0.05, iteration=8 has best effect，smaller has noise，bigger makes distortion
plt.figure()
plt.suptitle('closing operator with different iterations')
plt.subplot(221)
closing_mask(floor_plane_2D, 6)
plt.title('iteration=6')
plt.subplot(222)
closing_mask(floor_plane_2D, 7)
plt.title('iteration=7')
plt.subplot(223)
closing_mask(floor_plane_2D, 8)
plt.title('iteration=8')
plt.subplot(224)
closing_mask(floor_plane_2D, 9)
plt.title('iteration=9')
plt.show()

best_floor = closing_mask_noShow(floor_plane_2D, 8)
closing_mask(floor_plane_2D, 8)
plt.title('filted floor with closing iteration=8')
plt.show()


# find all pixels that not belong the floor
other_points_bool = ~floor_plane # shape: (202007,)
other_points = mask_image(cloud1_nonzeroz, other_points_bool) # reserve true, remove false, shape: (45746, 3)

# -------------------------------
# Step4. Finding the Top Plane of the Box
# using RANSAC to find the dominant plane within this except floor set
# note: when the threshold= 0.035, the number of feature without preprossing= 1
top_inlier, top_plane = ransac(other_points, 0.005, 1000)
print('max inliers of top plane:', top_inlier)

# remove points that not belong top plane from other points
top_plane_3D = mask_image(other_points, top_plane) # shape: (35000,3)
show_point_cloud(top_plane_3D, 4, '3D top plane with subsample 4')

# show 2D top plane
# firstly we filter the other point and reserve top plane
# The number of True in other_points_bool= the row number of top_plane
mask3 = visualize_mask(other_points_bool, top_plane)
# then we fill the array that should be same as the original shape, (217088, )
mask4 = visualize_mask(filted_cloud1, mask3)
top_plane_2D = np.reshape(mask4, mat1['distances1'].shape) # shape: (424,512)
show_image(top_plane_2D, '2D top plane')

# find the largest connected component in the mask with scipy's label
# optimal: number of features= 1 ?
labeled_array, num_features = scipy.ndimage.label(top_plane_2D)
print('number of features: ', num_features)

# try to use opening and closing operator
# Conclusion: processing will improve the result
# by comparing the processed images, I think opening operator with 3 iterations is better
# opening operator: remove the small regions (noise)
plt.figure()
plt.suptitle('labeled opening operator with different iterations')
plt.subplot(221)
label_opening(top_plane_2D, 1)
plt.title('iteration=1')
plt.subplot(222)
label_opening(top_plane_2D, 2)
plt.title('iteration=2')
plt.subplot(223)
label_opening(top_plane_2D, 3)
plt.title('iteration=3')
plt.subplot(224)
label_opening(top_plane_2D, 4)
plt.title('iteration=4')
plt.show()

plt.figure()
plt.suptitle('labeled closing operator with different iterations')
plt.subplot(221)
label_closing(top_plane_2D, 1)
plt.title('iteration=1')
plt.subplot(222)
label_closing(top_plane_2D, 2)
plt.title('iteration=2')
plt.subplot(223)
label_closing(top_plane_2D, 3)
plt.title('iteration=3')
plt.subplot(224)
label_closing(top_plane_2D, 4)
plt.title('iteration=4')
plt.show()

# show the processed top plane
# actually the goal is removing noise, so we can only use the opening
best_top = opening_mask_noShow(top_plane_2D, 3)
show_image(best_top, 'filted top with opening iteration=3')

# overlap the top plane on the 2D distance image of the floor plane
# distance image, different values mean different color
# so multiple any number to distinguish the top and floor
overlap_topfloor = 4 * best_top + best_floor
show_image(overlap_topfloor, 'overlaped image')


# ---------------------
# Step5. Measuring the Dimensions of the Box
# calculation the height of the box)
height = cal_height2(top_plane_3D, floor_plane_3D, 400)
print('height of the box= ', height)

# find the 3D-coordinates of the corners
# firstly I want to convert the top_best from 2D to 3D like top_plane_3D
best_top_1D = np.reshape(best_top, best_top.shape[0] * best_top.shape[:][1]) # best_top: (424, 512)
best_top_1D_3 = np.c_[best_top_1D, best_top_1D, best_top_1D]
# delete the false and generate a single column array which includes all the 3D top plane float data
best_top_3D = cloud1[best_top_1D_3]
best_top_3D = np.reshape(best_top_3D, (len(best_top_3D) // 3, 3))  # shape: (36100, 3)
show_point_cloud(best_top_3D, 3, 'best_top_3D')

# capture corners
# firstly we get six special points, which four of them should be the corners
corners1 = best_top_3D.argmax(axis=0)
corners2 = best_top_3D.argmin(axis=0)
pc0 = best_top_3D[corners1[0]]
pc1 = best_top_3D[corners1[1]]
pc2 = best_top_3D[corners1[2]]
pc3 = best_top_3D[corners2[0]]
pc4 = best_top_3D[corners2[1]]
pc5 = best_top_3D[corners2[2]]
list1 = [pc0, pc1, pc2, pc3, pc4, pc5]

# To find the four corners form the mentioned six points, we calculate the min distances each other
# And delete the two which have minimum distances
list_distance = []
for ii in range(5):
    kleinest_dis = smallestDis(list1[ii], list1[ii+1:])
    list_distance.append(kleinest_dis)
smallst_index = np.argsort(list_distance) # [2 0 3 1 4]
cornerPoint1 = list1[smallst_index[2]]
cornerPoint2 = list1[smallst_index[3]]
cornerPoint3 = list1[smallst_index[4]]
cornerPoint4 = list1[5]

# return the 3D corner points to the 2D overlapped image
point_find0 = returnProjection(smallst_index[2], corners1, corners2)
point_find1 = returnProjection(smallst_index[3], corners1, corners2)
point_find2 = returnProjection(smallst_index[4], corners1, corners2)
point_find3 = corners2[2] # the last point
mark_cornerfind = np.zeros(len(best_top_3D), dtype=bool) # all points in best top
mark_cornerfind[point_find0] = True
mark_cornerfind[point_find1] = True
mark_cornerfind[point_find2] = True
mark_cornerfind[point_find3] = True
mark_cornerfind_least = visualize_mask(best_top_1D, mark_cornerfind)
cornerFind_2dMatrix = np.reshape(mark_cornerfind_least, (424, 512)) # only cp True, others False
noZero_2d = np.nonzero(cornerFind_2dMatrix)
y_way = noZero_2d[0][:]
x_way = noZero_2d[1][:]

# 在2Dimage中展示那四个角点
plt.imshow(overlap_topfloor)
plt.title('Overlapping image with 4 corners')
plt.scatter(x_way, y_way, s=40, c='r')
plt.show()

# calculate the side length from every two points
long12 = np.sqrt(np.sum(np.square(cornerPoint1 - cornerPoint2)))
long13 = np.sqrt(np.sum(np.square(cornerPoint1 - cornerPoint3)))
long14 = np.sqrt(np.sum(np.square(cornerPoint1 - cornerPoint4)))
long23 = np.sqrt(np.sum(np.square(cornerPoint2 - cornerPoint3)))
long24 = np.sqrt(np.sum(np.square(cornerPoint2 - cornerPoint4)))
long34 = np.sqrt(np.sum(np.square(cornerPoint3 - cornerPoint4)))
# corresponding [2*short side, 2* long side, 2* diagonal side]
length = [long12, long13, long14, long23, long24, long34]
length.sort()
width1 = length[0]
width2 = length[1]
length1 = length[2]
length2 = length[3]
print('width of box= ', width1, ' and ', width2)
print('length of box=', length1, ' and ', length2)





















