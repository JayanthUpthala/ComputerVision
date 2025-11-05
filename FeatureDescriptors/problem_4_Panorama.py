'''
Created on Mar 10, 2023

@author: aelsayed
'''

import cv2
import numpy as np

def match_features(des1, des2):
    # FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# Warp img2 to img1 using the homography matrix H
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return output_img

if __name__=='__main__':
    
    #read images and prepare them
    img1 = cv2.imread('D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\scene\\street_1.jpg', 0)   
    img2 = cv2.imread('D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\scene\\street_2.jpg', 0)
    img3 = cv2.imread('D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\scene\\street_3.jpg', 0)

    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # Initialize the SURF detector
    sift = cv2.SIFT_create()

    # Extract the keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img3, None)

    
    good_matches_12 = match_features(des1, des2)
    good_matches_23 = match_features(des2, des3)

    min_match_count = 10

    # Stitch img1 and img2
    if len(good_matches_12) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_12]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_12]).reshape(-1, 1, 2)

        # Find Homography
        H12, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp images
        panorama12 = warpImages(img1, img2, H12)
    else:
        print("Not enough matches between img1 and img2.")
        

    # Find the Homography between the good matches keypoints
    if len(good_matches_23) > min_match_count:
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches_23]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp3[m.trainIdx].pt for m in good_matches_23]).reshape(-1, 1, 2)

        # Find Homography
        H23, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Transform the other images withrespect to the fixed reference one
        result = warpImages(panorama12, img3, H23)
        #result = cv2.equalizeHist(result)
        cv2.imshow('Panorama output', result)
        cv2.imwrite("D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\scene\\panorama_output.jpg", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Not enough matches between img2 and img3.")

