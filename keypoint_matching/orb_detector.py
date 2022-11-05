import cv2
import numpy as np

# -------------------------------
# --- PARAMS TO CHANGE IN UI ----
# -------------------------------

# Param: How many points to consider
maxpoints = 5

# Param: Search Perspective or Affine transform
searchMode = "affine"
# searchMode = "perspective"

# Param: Use RANSAC point filtering
MODE_RANSAC = True

# ------------------------------------
# --- END: PARAMS TO CHANGE IN UI ----
# ------------------------------------

DEBUG = False

SRC_FOLDER = '../pic/'
OUTPUT_FOLDER = '../data/output/'
RESULT_FOLDER = '../data/result/'

SOURCE_IMAGE1 = SRC_FOLDER + 'ford.png'
SOURCE_IMAGE2 = SRC_FOLDER + 'f3.jpg'

OUTPUT_IMAGE1 = OUTPUT_FOLDER + 'keypoints1.jpg'
OUTPUT_IMAGE2 = OUTPUT_FOLDER + 'keypoints2.jpg'

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

own_matches = []

for match in matches:
    qIdx = match.queryIdx
    tIdx = match.trainIdx
    own_matches.append((key_pts1[qIdx], key_pts2[tIdx]))

own_matches = np.array(own_matches).reshape(-1, 2, 2)
print("Found matches")
print(own_matches.shape)
h, w, _ = img2.shape

if maxpoints > own_matches.shape[0]:
    maxpoints = own_matches.shape[0]

# Elso kep illesztese a masodikra
if own_matches.shape[0] < 3:
    print("Not Enough point match to get transform")
    img_1to2 = np.zeros((h, w, 3))
else:
    if searchMode == "affine":
        match_points1 = np.array(own_matches[:maxpoints, 0, :]).reshape(-1, 2).astype(float)
        match_points2 = np.array(own_matches[:maxpoints, 1, :]).reshape(-1, 2).astype(float)
        if MODE_RANSAC:
            M = cv2.estimateAffine2D(match_points1, match_points2, method=cv2.RANSAC)
        else:
            M = cv2.estimateAffine2D(match_points1, match_points2)
        print("Affine Transformation matrix")
        print(M[0])
        img_1to2 = cv2.warpAffine(img1.copy(), M[0], (w, h))
    else:
        if own_matches.shape[0] < 4:
            print("Needs 4 matching points for Perspective transformation")
        if MODE_RANSAC:
            M, mask = cv2.findHomography(own_matches[:maxpoints,0,:], own_matches[:maxpoints,1,:], method=cv2.RANSAC)
        else:
            M, mask = cv2.findHomography(own_matches[:maxpoints,0,:], own_matches[:maxpoints,1,:])
        print("Perspective Transformation matrix")
        print(M)
        img_1to2 = cv2.warpPerspective(img1.copy(), M, (w, h))

cv2.imshow('input', img2)
cv2.imshow('result', img_1to2)

if DEBUG:
    img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches[:maxpoints],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('draw_matches', img3)

cv2.waitKey()

