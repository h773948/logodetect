import cv2
import numpy as np

SOURCE_IMAGE1='../pic/ford.png'
SOURCE_IMAGE2='../pic/ford2.jpg'

OUTPUT_IMAGE1='keypoints1.jpg'
OUTPUT_IMAGE2='keypoints2.jpg'

MATCHING_IMAGE='flann_matching.jpg'

## képek beolvasása
img1 = cv2.imread(SOURCE_IMAGE1);
img2 = cv2.imread(SOURCE_IMAGE2);

## a képet szürkeárnyalatossá konvertáljuk
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

## jellemzőpontok detektálása 
#orb = cv2.ORB_create()
#keypoints1 = orb.detect(gray_img1)
#keypoints2 = orb.detect(gray_img2)


## kulcspont leírók számítása
#keypoints1, descriptors1 = orb.compute(gray_img1, keypoints1)
#keypoints2, descriptors2 = orb.compute(gray_img2, keypoints2)

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

matches = flann.knnMatch(np.asarray(descriptors1,np.float32),np.asarray(descriptors2,np.float32),k=2) ## ez most kNN-alapú, 
                                                        ## minden pontnak két lehetséges párja lehet

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

#img3 = cv2.drawMatches(        img1,keypoints1,       img2,keypoints2,      matches[:50], None, flags=2)
img3 = cv2.drawMatchesKnn(img1,keypoints1,img2,keypoints2,matches,None,**draw_params)
cv2.imwrite(MATCHING_IMAGE, img3)

