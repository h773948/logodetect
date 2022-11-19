import cv2
import numpy as np

# -------------------------------
# --- PARAMS TO CHANGE IN UI ----
# -------------------------------

# Param: How many points to consider
maxpoints = 10

# Param: Search Perspective or Affine transform
searchMode = "affine"
# searchMode = "perspective"

# Param: Use RANSAC point filtering
MODE_RANSAC = True

# Param: Deterimes to draw bounding box ot not
DRAW_BOUNDING_BOX = True

# ------------------------------------
# --- END: PARAMS TO CHANGE IN UI ----
# ------------------------------------
SRC_FOLDER = '../pic/'
SOURCE_IMAGE1_NAME = 'ford.png'
SOURCE_IMAGE2_NAME = 'ford2.jpg'
# SOURCE_IMAGE2_NAME = '224886291.jpg'
# SOURCE_IMAGE2_NAME = '255740214.jpg'
SOURCE_IMAGE1 = SRC_FOLDER + SOURCE_IMAGE1_NAME
SOURCE_IMAGE2 = SRC_FOLDER + SOURCE_IMAGE2_NAME


def o_d(maxpoints,searchMode,MODE_RANSAC,DRAW_BOUNDING_BOX,SOURCE_IMAGE1,SOURCE_IMAGE2):
    DEBUG = False

    
    OUTPUT_FOLDER = '../data/output/'
    RESULT_FOLDER = '../data/result/'
    ANNOTATIONS = '../data/train/logo_box_annotations.txt'

    
    

    

    OUTPUT_IMAGE1 = OUTPUT_FOLDER + 'keypoints1.jpg'
    OUTPUT_IMAGE2 = OUTPUT_FOLDER + 'keypoints2.jpg'

    with open(ANNOTATIONS, 'r') as reader:
        line = reader.readline()
        while line != '':  # The EOF char is an empty string
            line = reader.readline()
            line_data = line.split(' ')
            if line_data[0] == SOURCE_IMAGE2_NAME:
                break

    train_class = None
    train_bounding_box = None
    if line == '':
        print("No annotation found for this file")
    else:
        train_class = line_data[2]
        train_bounding_box = [
            [int(line_data[3]), int(line_data[4])],
            [int(line_data[5]), int(line_data[6])]
        ]
        print("Bounding Box: " + str(train_bounding_box))

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

    # For determining calculated output bounding box (the red one)
    transform_box = None
    img1_box = np.array([
        [0,             0,              1],
        [img1.shape[1], 0,              1],
        [img1.shape[1], img1.shape[0],  1],
        [0,             img1.shape[0],  1],
    ]).reshape(-1,3)

    def get_brect_fomr_points(transform_box):
        if (transform_box.shape[1] == 3): # if perspective
            for vect in transform_box:
                vect[0] = vect[0] / vect[2]
                vect[1] = vect[1] / vect[2]
        tr_brect = transform_box[:, :2]
        # print(tr_brect)
        tr_brect = cv2.boundingRect(tr_brect.astype(int))
        tr_bound_rect = np.ndarray((2, 2), dtype=int)
        tr_bound_rect[0] = tr_brect[0], tr_brect[1]
        tr_bound_rect[1] = tr_brect[0] + tr_brect[2], tr_brect[1] + tr_brect[3]
        return tr_bound_rect

    def run_transform_finding():
        global maxpoints
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

                # Get the red bounding box
                transform_box = np.transpose(np.matmul(M[0], np.transpose(img1_box)))
                transform_box = get_brect_fomr_points(transform_box)

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

                transform_box = np.transpose(np.matmul(M, np.transpose(img1_box)))
                transform_box = get_brect_fomr_points(transform_box)

        if DRAW_BOUNDING_BOX and train_bounding_box is not None:
            img_1to2 = cv2.rectangle(img_1to2, (train_bounding_box[0]), (train_bounding_box[1]), (0, 255, 0), 3)
        if DRAW_BOUNDING_BOX:
            img_1to2 = cv2.rectangle(img_1to2, (transform_box[0]), (transform_box[1]), (0, 0, 255), 3)

        cv2.imshow('input', img2)
        cv2.imshow('result', img_1to2)

    # If params changed run this again
    run_transform_finding()

    if DEBUG:
        img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches[:maxpoints],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('draw_matches', img3)

    cv2.waitKey()

o_d(maxpoints, searchMode, MODE_RANSAC, DRAW_BOUNDING_BOX,SOURCE_IMAGE1,SOURCE_IMAGE2)