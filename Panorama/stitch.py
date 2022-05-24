import cv2

img1 = cv2.imread('IMG_1805.JPG' )
img1 = cv2.resize(img1, (0,0), None, 0.4, 0.4)
img2 = cv2.imread('IMG_1806.JPG' )
img2 = cv2.resize(img2, (0,0), None, 0.4, 0.4)
img3 = cv2.imread('IMG_1807.JPG' )
img3 = cv2.resize(img3, (0,0), None, 0.4, 0.4)
images = [img1, img2, img3]
stitcher = cv2.Stitcher.create()
(status, result) = stitcher.stitch(images)
cv2.imshow('x',result)
cv2.waitKey(0)