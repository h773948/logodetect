import cv2
import numpy as np

# -------------------------------
# --- PARAMS TO CHANGE IN UI ----
# -------------------------------

# Param: Lowe Ratio: default from the papers: 0.7
lowe_ratio = 0.7

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
SOURCE_IMAGE2 = SRC_FOLDER + 'f4.jpg'

OUTPUT_IMAGE1 = OUTPUT_FOLDER + 'keypoints1.jpg'
OUTPUT_IMAGE2 = OUTPUT_FOLDER + 'keypoints2.jpg'

MATCHING_IMAGE= RESULT_FOLDER + 'flann_matching.jpg'

## képek beolvasása
img1 = cv2.imread(SOURCE_IMAGE1)
img2 = cv2.imread(SOURCE_IMAGE2)

## a képet szürkeárnyalatossá konvertáljuk
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

## jellemzőpontok detektálása
sift = cv2.SIFT_create()
keypoints1 = sift.detect(gray_img1, None)
keypoints2 = sift.detect(gray_img2, None)

## kulcspont leírók számítása
keypoints1, descriptors1 = sift.compute(gray_img1, keypoints1)
keypoints2, descriptors2 = sift.compute(gray_img2, keypoints2)

key_pts1 = np.array([key_point.pt for key_point in keypoints1]).reshape(-1, 1, 2)
key_pts2 = np.array([key_point.pt for key_point in keypoints2]).reshape(-1, 1, 2)

## pontpárok keresése
# FLANN parameterek
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

distances = np.array([(m1.distance, m2.distance) for (m1, m2) in knn_matches])
trainIdxS = np.array([(m1.trainIdx , m2.trainIdx ) for (m1, m2) in knn_matches])

print(distances.shape)
print(trainIdxS.shape)

own_matches = []

# Lowe-féle arányteszt
for i, (m, n) in enumerate(knn_matches):
    if m.distance < lowe_ratio * n.distance:
        own_matches.append([keypoints1[i], keypoints2[m.trainIdx]])

own_matches = np.array([(kp1.pt, kp2.pt) for (kp1, kp2) in own_matches])
print("Matched points:")
print(own_matches.shape)
h, w, _ = img2.shape

# Elso kep illesztese a masodikra
if own_matches.shape[0] < 3:
    print("Not Enough point match to get transform")
    img_1to2 = np.zeros((h, w, 3))
else:
    if searchMode == "affine":
        match_points1 = np.array(own_matches[:, 0, :]).reshape(-1, 2).astype(float)
        match_points2 = np.array(own_matches[:, 1, :]).reshape(-1, 2).astype(float)
        if MODE_RANSAC:
            M = cv2.estimateAffine2D(match_points1, match_points2, method=cv2.RANSAC)
        else:
            M = cv2.estimateAffine2D(match_points1, match_points2)
        print("Affine Transformation matrix")
        if M[0] is None:
            print("NOT FOUND: Affine transformation")
            img_1to2 = np.zeros((h, w, 3))
        else:
            print(M[0])
            img_1to2 = cv2.warpAffine(img1.copy(), M[0], (w, h))
    else:
        if own_matches.shape[0] < 4:
            print("Needs 4 matching points for Perspective transformation")
        if MODE_RANSAC:
            M, mask = cv2.findHomography(own_matches[:, 0, :], own_matches[:, 1, :], method=cv2.RANSAC)
        else:
            M, mask = cv2.findHomography(own_matches[:, 0, :], own_matches[:, 1, :])
        print("Perspective Transformation matrix")
        print(M)
        img_1to2 = cv2.warpPerspective(img1.copy(), M, (w, h))

cv2.imshow('input', img2)
cv2.imshow('result', img_1to2)

if DEBUG:
    # Csak a jó párosítások érdekelnek bennünket, így azokat maszkoljuk
    matchesMask = [[0,0] for i in range(len(knn_matches))]
    # Lowe-féle arányteszt
    for i,(m,n) in enumerate(knn_matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, knn_matches[:10], None)
    cv2.imshow('draw_matches', img3)

cv2.waitKey()

