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
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    queue_indices = []
    skipped_indices = []
    overlap_arr = np.ones((N, N), dtype=np.uint8)

    if any(e is None for e in imgs):
        print("Any one of the images is not found, returning zeros overlap array.")
        overlap_arr = np.zeros((N, N), dtype=np.uint8)
        return overlap_arr

    def merge_images(img1, img2, index, append_if_skip=True):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kps1, des1 = sift.detectAndCompute(gray1, None)
        kps2, des2 = sift.detectAndCompute(gray2, None)

        matches = bf.match(des1, des2)
        best_matches = sorted(matches, key=lambda x: x.distance)[0:len(matches)//3]

        kp_pts1 = np.float32([kp.pt for kp in kps1])
        kp_pts2 = np.float32([kp.pt for kp in kps2])
        pts1 = np.float32([kp_pts1[m.queryIdx] for m in best_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp_pts2[m.trainIdx] for m in best_matches]).reshape(-1, 1, 2)

        # best_matches = []
        # for queryIdx, queryMatrix in enumerate(des1):
        #     min_dist = np.inf
        #     best_indices = [0, 0]
        #     for trainIdx, trainMatrix in enumerate(des2):
        #         distance = np.linalg.norm(queryMatrix - trainMatrix)
        #         if (distance < min_dist):
        #             min_dist = distance
        #             best_indices = [queryIdx, trainIdx, distance]
        #     best_matches.append(best_indices)
        # best_matches = sorted(best_matches, key=lambda x: x[2])[0:400]
        #
        # kp_pts1 = np.float32([kp.pt for kp in kps1])
        # kp_pts2 = np.float32([kp.pt for kp in kps2])
        # pts1 = np.float32([kp_pts1[m[0]] for m in best_matches]).reshape(-1, 1, 2)
        # pts2 = np.float32([kp_pts2[m[1]] for m in best_matches]).reshape(-1, 1, 2)

        (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        rot_matrix = np.array(np.round(H, decimals=1))[:2, :2]
        rot_matrix_0011 = np.abs(rot_matrix[0, 0]) == np.abs(rot_matrix[1, 1])
        rot_matrix_0110 = np.abs(rot_matrix[0, 1]) == np.abs(rot_matrix[1, 0])
        if (not (rot_matrix_0011 and rot_matrix_0110)):
            if(append_if_skip):
                print(f"{imgmark}_{index+1}.png could not match with stitched image, adding to queue!")
                queue_indices.append(index)
            else:
                print(f"Did not match again, skipping {imgmark}_{index+1}.png!")
                skipped_indices.append(index)
            return img1

        img1_shape = img1.shape[0:2]  # src
        img2_shape = img2.shape[0:2]  # des

        unwarped_bounds = np.array([[0, 0], [img1_shape[1], 0], [0, img1_shape[0]], [img1_shape[1], img1_shape[0]]])
        homogenous_unwarped_bounds = []
        for x, y in unwarped_bounds:
            homogenous_unwarped_bounds.append([x, y, 1])
        homogenous_unwraped_bound = np.array(homogenous_unwarped_bounds).squeeze()
        homogeneous_warped_bounds = []
        for x, y, z in homogenous_unwraped_bound:
            homogeneous_warped_bounds.append(np.dot(H, np.array([[x], [y], [z]])).T)
        homogeneous_warped_bounds = np.array(homogeneous_warped_bounds).squeeze()
        warped_bounds = []
        for x, y, z in homogeneous_warped_bounds:
            warped_bounds.append([x / z, y / z])
        warped_bounds = np.array(warped_bounds).squeeze()
        min_warped_bounds = np.min(warped_bounds, axis=0)
        max_warped_bounds = np.max(warped_bounds, axis=0)

        des_bounds = np.array([[0, 0], [img2_shape[1], 0], [0, img2_shape[0]], [img2_shape[1], img2_shape[0]]])
        min_des_bounds = np.min(des_bounds, axis=0)
        max_des_bounds = np.max(des_bounds, axis=0)

        min_output_bounds = (np.min([min_warped_bounds[0], min_des_bounds[0]]), np.min([min_warped_bounds[1], min_des_bounds[1]]))
        max_output_bounds = (np.max([max_warped_bounds[0], max_des_bounds[0]]), np.max([max_warped_bounds[1], max_des_bounds[1]]))

        offset = np.array([[1, 0, -min_output_bounds[0]], [0, 1, -min_output_bounds[1]], [0, 0, 1]])
        merged_img = cv2.warpPerspective(img1, np.dot(offset, H), (round(max_output_bounds[0] - min_output_bounds[0]), round(max_output_bounds[1] - min_output_bounds[1])))
        merged_img[int(offset[1, 2]): img2_shape[0] + int(offset[1, 2]), int(offset[0, 2]): img2_shape[1] + int(offset[0, 2])] = img2

        # Removing black borders from rows and cols
        merged_img = np.delete(merged_img, np.where(np.sum(merged_img[:, :, 0], axis=1) == 0), axis=0)
        merged_img = np.delete(merged_img, np.where(np.sum(merged_img[:, :, 0], axis=0) == 0), axis=1)
        return merged_img

    output_img = imgs[0]

    # Merging first image with others
    for i in range(1, N):
        output_img = merge_images(output_img, imgs[i], i)
        # cv2.imshow("output", output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    if(queue_indices):
        print("Trying skipped images again.")
        # Merging skipped images again with final output
        for i in queue_indices:
            output_img = merge_images(output_img, imgs[i], i, append_if_skip=False)
            # cv2.imshow("output", output_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    for i in skipped_indices:
        overlap_arr[:, i] = np.zeros_like(overlap_arr[:, i])
        overlap_arr[i, :] = np.zeros_like(overlap_arr[:, i])
        overlap_arr[i, i] = 1

    cv2.imwrite(savepath, output_img)
    print(f"Stitched image {savepath} saved!")

    return overlap_arr


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)

    # bonus
    overlap_arr2 = stitch('t3', N=4, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
