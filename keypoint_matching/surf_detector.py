import cv2
import numpy as np

MAXPOINTS = 5
DEBUG = False

SRC_FOLDER = '../pic/'
OUTPUT_FOLDER = '../data/output/'
RESULT_FOLDER = '../data/result/'

SOURCE_IMAGE1 = SRC_FOLDER + 'ford.png'
SOURCE_IMAGE2 = SRC_FOLDER + 'f12.jpg'

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

own_mathes = []

# Lowe-féle arányteszt
for i, (m, n) in enumerate(knn_matches):
    if m.distance < 0.7 * n.distance:
        own_mathes.append([keypoints1[i], keypoints2[m.trainIdx]])

own_mathes2 = np.array([(kp1.pt, kp2.pt) for (kp1, kp2) in own_mathes])
print("Matched points:")
print(own_mathes2.shape)

# Elso kep illesztese a masodikra
M, mask = cv2.findHomography(own_mathes2[:,0,:], own_mathes2[:,1,:], cv2.LMEDS)
print("Transformation Matrix")
print(M)
h, w, _ = img2.shape
img_1to2 = cv2.warpPerspective(img1.copy(), M, (w, h))


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

# img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None)

# cv2.imwrite(MATCHING_IMAGE, img3)

cv2.imshow('asdasdasd', img2)
# cv2.imshow('draw_matches', img3)
cv2.imshow('result', img_1to2)
cv2.waitKey()

