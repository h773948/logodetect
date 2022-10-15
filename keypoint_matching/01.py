import cv2
import numpy as np

SRC_FOLDER = '../pic/'
OUTPUT_FOLDER = '../data/output/'
RESULT_FOLDER = '../data/result/'

SOURCE_IMAGE1 = SRC_FOLDER + 'ford.png'
SOURCE_IMAGE2 = SRC_FOLDER + 'ford2.jpg'

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
orb = cv2.ORB_create()
keypoints1 = orb.detect(gray_img1)
keypoints2 = orb.detect(gray_img2)

## kulcspont leírók számítása
keypoints1, descriptors1 = orb.compute(gray_img1, keypoints1)
keypoints2, descriptors2 = orb.compute(gray_img2, keypoints2)

## jellemzőpontok detektálása
surf = cv2.SIFT_create()
keypoints1 = surf.detect(gray_img1, None)
keypoints2 = surf.detect(gray_img2, None)

## kulcspont leírók számítása
keypoints1, descriptors1 = surf.compute(gray_img1, keypoints1)
keypoints2, descriptors2 = surf.compute(gray_img2, keypoints2)

## kulcspontok kirajzolása
out_img1 = cv2.drawKeypoints(gray_img1, keypoints1, descriptors1, color=(255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
out_img2 = cv2.drawKeypoints(gray_img2, keypoints2, descriptors2, color=(255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite(OUTPUT_IMAGE1, out_img1)
cv2.imwrite(OUTPUT_IMAGE2, out_img2)

## pontpárok keresése
# FLANN parameterek
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(descriptors1, descriptors2, k=2) ## ez most kNN-alapú,
                                                        ## minden pontnak két lehetséges párja lehet

pts1 = np.array([key_point.pt for key_point in keypoints1]).reshape(-1, 1, 2)
pts2 = np.array([key_point.pt for key_point in keypoints2]).reshape(-1, 1, 2)

distances = np.array([(m1.distance, m2.distance) for (m1, m2) in matches])
print(distances)

trainIdxS = np.array([(m1.trainIdx , m2.trainIdx ) for (m1, m2) in matches])
print(trainIdxS)

asd = []

# Lowe-féle arányteszt
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        asd.append([keypoints1[i], keypoints2[m.trainIdx]])

asd2 = np.array([(kp1.pt, kp2.pt) for (kp1, kp2) in asd])
print(asd2[:,0,:])

# Elso kep illesztese a masodikra
M, mask = cv2.findHomography(asd2[:,0,:], asd2[:,1,:], cv2.RANSAC)
print(M)
h, w, _ = img2.shape
img_1to2 = cv2.warpPerspective(img1.copy(), M, (w, h))

print(distances.shape)
print(trainIdxS.shape)

# Csak a jó párosítások érdekelnek bennünket, így azokat maszkoljuk
matchesMask = [[0,0] for i in range(len(matches))]

# Lowe-féle arányteszt
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

# img3 = cv2.drawMatches(        img1,keypoints1,       img2,keypoints2,      matches[:50], None, flags=2)
# img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None)
# cv2.imwrite(MATCHING_IMAGE, img3)

cv2.imshow('asdasdasd', img2)
cv2.imshow('result', img_1to2)
cv2.waitKey()

