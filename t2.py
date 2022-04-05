# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# TODO remove
np.set_printoptions(suppress=True)


def stitch(imgmark, N=4, savepath=''):  # For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N + 1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)

    "Start you code here"

    sift = cv2.xfeatures2d.SIFT_create()
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    output_img = imgs[0]

    for i in range(1, N):
        gray1 = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        kps1, des1 = sift.detectAndCompute(gray1, None)
        gray2 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        kps2, des2 = sift.detectAndCompute(gray2, None)

        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        # print(np.column_stack((np.array([m.queryIdx for m in matches[0:20]]), np.array([m.trainIdx for m in matches[0:20]]))))

        best_matches = []
        for queryIdx, queryMatrix in enumerate(des1):
            min_dist = np.inf
            best_indices = [0, 0]
            for trainIdx, trainMatrix in enumerate(des2):
                distance = np.linalg.norm(queryMatrix - trainMatrix)
                if(distance < min_dist):
                    min_dist = distance
                    best_indices = [queryIdx, trainIdx, distance]
            best_matches.append(best_indices)
        best_matches = sorted(best_matches, key=lambda x: x[2])[0:400]

        kp_pts1 = np.float32([kp.pt for kp in kps1])
        kp_pts2 = np.float32([kp.pt for kp in kps2])
        pts1 = np.float32([kp_pts1[m[0]] for m in best_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp_pts2[m[1]] for m in best_matches]).reshape(-1, 1, 2)

        (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        rot_matrix = np.array(np.round(H, decimals=1))[:2, :2]
        # print(rot_matrix)

        # Checks if elements follow rotation matrix sin cos sequence
        rot_matrix_0011 = np.abs(rot_matrix[0, 0]) == np.abs(rot_matrix[1, 1])
        rot_matrix_0110 = np.abs(rot_matrix[0, 1]) == np.abs(rot_matrix[1, 0])
        if (not (rot_matrix_0011 and rot_matrix_0110)):
            print(f"Stitched Image and Image{i} could not match, skipping!")
            continue

        # output_img = warpTwoImages(imgs[i], output_img,  H)
        h1, w1 = imgs[i].shape[0], imgs[i].shape[1]
        h2, w2 = output_img.shape[0], output_img.shape[1]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(np.min(pts, axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(np.max(pts, axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

        output_img = cv2.warpPerspective(output_img, Ht.dot(H), (xmax - xmin, ymax - ymin))
        output_img[t[1]:h1 + t[1], t[0]:w1 + t[0]] = imgs[i]
        cv2.imshow("output", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    overlap_arr = np.array([])
    return overlap_arr


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)

    # bonus
    # overlap_arr2 = stitch('t3', savepath='task3.png')
    # with open('t3_overlap.txt', 'w') as outfile:
    #     json.dump(overlap_arr2.tolist(), outfile)
