import cv2
import numpy as np

MAXPOINTS = 5
DEBUG = False

SRC_FOLDER = '../pic/'
OUTPUT_FOLDER = '../data/output/'
RESULT_FOLDER = '../data/result/'

SOURCE_IMAGE1 = SRC_FOLDER + 'ford.png'
SOURCE_IMAGE2 = SRC_FOLDER + 'f2.jpg'

OUTPUT_IMAGE1 = OUTPUT_FOLDER + 'keypoints1.jpg'
OUTPUT_IMAGE2 = OUTPUT_FOLDER + 'keypoints2.jpg'

MATCHING_IMAGE= RESULT_FOLDER + 'flann_matching.jpg'

## képek beolvasása
img1 = cv2.imread(SOURCE_IMAGE1)
img2 = cv2.imread(SOURCE_IMAGE2)

## a képet szürkeárnyalatossá konvertáljuk
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
keypoints1 = orb.detect(gray_img1, None)
keypoints2 = orb.detect(gray_img2, None)

## kulcspont leírók számítása
keypoints1, descriptors1 = orb.compute(gray_img1, keypoints1)
keypoints2, descriptors2 = orb.compute(gray_img2, keypoints2)

key_pts1 = np.array([key_point.pt for key_point in keypoints1]).reshape(-1, 1, 2)
key_pts2 = np.array([key_point.pt for key_point in keypoints2]).reshape(-1, 1, 2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(descriptors1, descriptors2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

qeryIdxes = np.array([m1.queryIdx for m1 in matches])
trainIdxes = np.array([m1.trainIdx for m1 in matches])

key_pts1 = np.array([key_point.pt for key_point in keypoints1]).reshape(-1, 1, 2)
key_pts2 = np.array([key_point.pt for key_point in keypoints2]).reshape(-1, 1, 2)

asd = []

for match in matches:
    qIdx = match.queryIdx
    tIdx = match.trainIdx
    asd.append((key_pts1[qIdx], key_pts2[tIdx]))

asd = np.array(asd).reshape(-1, 2, 2)
print("Found matches")
print(asd.shape)

# Elso kep illesztese a masodikra
M, mask = cv2.findHomography(asd[:,0,:MAXPOINTS], asd[:,1,:MAXPOINTS], cv2.RANSAC)
print("Transformation matrix")
print(M)
h, w, _ = img2.shape
img_1to2 = cv2.warpPerspective(img1.copy(), M, (w, h))
if DEBUG:
    img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches[:MAXPOINTS],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('draw_matches', img3)

cv2.imshow('asdasdasd', img2)

cv2.imshow('result', img_1to2)
cv2.waitKey()

