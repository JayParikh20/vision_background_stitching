# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    sift = cv2.xfeatures2d.SIFT_create()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kps1, des1 = sift.detectAndCompute(gray1, None)
    kps2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
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
        warped_bounds.append([x/z, y/z])
    warped_bounds = np.array(warped_bounds).squeeze()
    min_warped_bounds = np.min(warped_bounds, axis=0)
    max_warped_bounds = np.max(warped_bounds, axis=0)

    des_bounds = np.array([[0, 0], [img2_shape[1], 0], [0, img2_shape[0]], [img2_shape[1], img2_shape[0]]])
    min_des_bounds = np.min(des_bounds, axis=0)
    max_des_bounds = np.max(des_bounds, axis=0)

    min_output_bounds = (np.min([min_warped_bounds[0], min_des_bounds[0]]), np.min([min_warped_bounds[1], min_des_bounds[1]]))
    max_output_bounds = (np.max([max_warped_bounds[0], max_des_bounds[0]]), np.max([max_warped_bounds[1], max_des_bounds[1]]))

    offset = np.array([[1, 0, -min_output_bounds[0]], [0, 1, -min_output_bounds[1]], [0, 0, 1]])
    transformed_img = cv2.warpPerspective(img1, np.dot(offset, H), (round(max_output_bounds[0] - min_output_bounds[0]), round(max_output_bounds[1] - min_output_bounds[1])))
    merged_img = transformed_img.copy()
    merged_img[int(offset[1, 2]): img2_shape[0] + int(offset[1, 2]), int(offset[0, 2]): img2_shape[1] + int(offset[0, 2])] = img2

    # finding intersected area of two images
    bw1 = np.zeros(gray1.shape, dtype=np.uint8)
    bw1.fill(1)
    bw2 = np.zeros(gray2.shape, dtype=np.uint8)
    bw2.fill(1)

    merged_bw = cv2.warpPerspective(bw1, np.dot(offset, H), (round(max_output_bounds[0] - min_output_bounds[0]), round(max_output_bounds[1] - min_output_bounds[1])))
    merged_bw[int(offset[1, 2]): img2_shape[0] + int(offset[1, 2]), int(offset[0, 2]): img2_shape[1] + int(offset[0, 2])] = merged_bw[int(offset[1, 2]): img2_shape[0] + int(offset[1, 2]), int(offset[0, 2]): img2_shape[1] + int(offset[0, 2])] + bw2

    cropped_img1 = transformed_img.copy()
    cropped_img1[np.where(merged_bw != 2)] = 0
    cropped_merged_img = merged_img.copy()
    cropped_merged_img[np.where(merged_bw != 2)] = 0

    # Subtracting common img1 part from common merged image
    cropped_diff = cv2.subtract(cropped_img1, cropped_merged_img)
    cropped_diff_img = cv2.cvtColor(cropped_diff, cv2.COLOR_BGR2GRAY)
    ret, diff_mask = cv2.threshold(cropped_diff_img, 0, 255, cv2.THRESH_OTSU)

    # Eroding and dilating pixels, removes small differences and dilates major ones
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((15, 15), np.uint8)
    img_erosion = cv2.erode(diff_mask, kernel_erode, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel_dilate, iterations=5)
    img_dilation[np.where(merged_bw != 2)] = 0
    ret, cleaned_mask = cv2.threshold(img_dilation, 0, 255, cv2.THRESH_BINARY)

    # Removing parts that are not needed
    cropped_img1[cleaned_mask == 0] = 0
    merged_img[cleaned_mask != 0] = 0
    output_img = cv2.add(cropped_img1, merged_img)

    cv2.imwrite(savepath, output_img)

    # cv2.imshow("output", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

