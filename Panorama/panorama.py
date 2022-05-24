import cv2
import numpy as np

"""
overview:
1. imread the three images, and add a copy of central that offset to right, then convert them to grayscale
2. extract SIFT keypoints and descriptors of four images
3. selecting the good matches that the distance < 0.5 * median
4. compute the optimal homography using ransac algorithm
5. obtain warpPerspective images
6. make mask and averaging the three image, make it uniform brightness
7. stitching them directly and crop the black frame
"""


# https://blog.csdn.net/EchoooZhang/article/details/105275661
def cvshow(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


# Exercise 2.1:
#
# Extract & match SIFT Keypoints and Descriptors
def sift_kp(image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)  # extract SIFT keypoints and descriptors
    return kp, des


# create a matching with the anchor points
# des1=anchor points, des2=matching points
def get_good_match(des1, des2):
    bf = cv2.BFMatcher()  # create the BFMatcher object
    matches = bf.match(des1, des2)  # returns the best match
    matches = sorted(matches, key=lambda x: x.distance)
    good = []
    dis = []
    for m in matches:
        dis.append(m.distance)
    dis_med = np.median(dis)
    for m in matches:
        if m.distance < 0.5 * dis_med:  # if alpha=0.1, there are no matches
            good.append(m)
    return good


# Exercise2.2: RANSAC Algorithm for Homography Computation
def ransac(img_src, img_des):
    kp1, des1 = sift_kp(img_src)
    kp2, des2 = sift_kp(img_des)
    goodMatch = get_good_match(des1, des2)
    MIN_MATCH_COUNT = 10
    # compute the homography
    if len(goodMatch) > MIN_MATCH_COUNT:
        # obtain the coordinates of the matched points in source image and destination image
        src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4  # select 4 random matches
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold, maxIters=10000)
        return H
    else:
        print('Not enough matches are found', (len(goodMatch) / MIN_MATCH_COUNT))


# create mask
def mask(image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if image[i, j, 0] != 0 and image[i, j, 1] != 0 and image[i, j, 2] != 0:
                mask[i, j] = 1.8
    mask = np.array(mask)
    sum_mask = np.sum(mask)
    cvshow('mask', mask)
    return sum_mask


def sumimage(image):
    image = np.array(image)
    sum_image = np.sum(image)
    return sum_image


def averaging(image):
    sum_mask = mask(image)
    sum_image = sumimage(image)
    average_image = image / (sum_image / sum_mask)
    cvshow('averaing', average_image)
    return average_image


# Exercise 2.3: Create the Panorama
def warp_pes(img_right_input, img_left_input, name):
    H = ransac(img_right_input, img_left_input)
    res = cv2.warpPerspective(img_right_input, H,
                              (3 * img_central.shape[1], img_central.shape[0]))
    cv2.imshow(name, res)
    cv2.waitKey(0)
    return res


def stitching(t1, t2, t3):
    t1[0:img_left.shape[0], img_left.shape[1]:-img_left.shape[1]] = \
        t2[0:img_left.shape[0], img_central.shape[1]:-img_central.shape[1]]
    t1[0:img_left.shape[0], 2 * img_left.shape[1]:] = \
        t3[0:img_left.shape[0], 2 * img_left.shape[1]:]
    return t1


def crop(frame):
    if not np.sum(frame[0]):
        return crop(frame[1:])
    if not np.sum(frame[-1]):
        return crop(frame[:-2])
    if not np.sum(frame[:, 0]):
        return crop(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return crop(frame[:, :-2])
    return frame


img_left = cv2.imread(r'harbor\IMG_1807.JPG')
img_central = cv2.imread(r'harbor\IMG_1806.JPG')
img_right = cv2.imread(r'harbor\IMG_1805.JPG')

img_left = cv2.resize(img_left, (0, 0), fx=0.2, fy=0.3)
img_central = cv2.resize(img_central, (0, 0), fx=0.2, fy=0.3)
img_central_copy = cv2.copyMakeBorder(img_central, 0, 0, img_central.shape[1], 0, cv2.BORDER_CONSTANT)
img_right = cv2.resize(img_right, (0, 0), fx=0.2, fy=0.3)

img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
img_central_gray = cv2.cvtColor(img_central, cv2.COLOR_BGR2GRAY)
img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
img_central_copy_gray = cv2.cvtColor(img_central_copy, cv2.COLOR_BGR2GRAY)

kp_left, des_left = sift_kp(img_left)
kp_central, des_central = sift_kp(img_central)
kp_right, des_right = sift_kp(img_right)
kp_central_copy, des_central_copy = sift_kp(img_central_copy)

draw_kp_left = cv2.drawKeypoints(img_left_gray, kp_left, None)
draw_kp_central = cv2.drawKeypoints(img_central_gray, kp_central, None)
draw_kp_right = cv2.drawKeypoints(img_right_gray, kp_right, None)
draw_kp_central_copy = cv2.drawKeypoints(img_central_copy_gray, kp_central_copy, None)

cvshow('img_left', np.hstack((img_left, draw_kp_left)))
cvshow('img_central', np.hstack((img_central, draw_kp_central)))
cvshow('img_right', np.hstack((img_right, draw_kp_right)))
cvshow('img_central_copy', np.hstack((img_central_copy, draw_kp_central_copy)))

goodMatch1 = get_good_match(des_central_copy, des_left)
draw_parameter = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
img_matched1 = cv2.drawMatches(img_left, kp_left, img_central_copy, kp_central_copy, goodMatch1, None, **draw_parameter)
cvshow('left matches anchor', img_matched1)

# central matches copy of central leads to none matches
goodMatch2 = get_good_match(des_central_copy, des_central)
draw_parameter = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
img_matched2 = cv2.drawMatches(img_central, kp_central, img_central_copy, kp_central_copy, goodMatch2, None,
                               **draw_parameter)
cvshow('central matches anchor', img_matched2)

goodMatch3 = get_good_match(des_central_copy, des_left)
draw_parameter = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
img_matched3 = cv2.drawMatches(img_right, kp_right, img_central_copy, kp_central_copy, goodMatch3, None,
                               **draw_parameter)
cvshow('right matches anchor', img_matched3)

t1 = warp_pes(img_left, img_central_copy, 'warpPerspective_left')
t2 = cv2.copyMakeBorder(img_central, 0, 0, img_central.shape[1], img_central.shape[1], cv2.BORDER_CONSTANT)
cvshow('warpPerspective_central', t2)
t3 = warp_pes(img_right, img_central_copy, 'warpPerspective_right')

t1 = averaging(t1)
t2 = averaging(t2)
t3 = averaging(t3)

panorama = stitching(t1, t2, t3)
panorama = crop(panorama)
cvshow('panorama', panorama)
