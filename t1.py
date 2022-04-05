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
    # gray1_kps = cv2.drawKeypoints(gray1, kps1, gray1)
    # cv2.imwrite('img1_kps.jpg', gray1_kps)

    kps2, des2 = sift.detectAndCompute(gray2, None)
    # gray2_kps = cv2.drawKeypoints(gray2, kps2, gray2)
    # cv2.imwrite('img2_kps.jpg', gray2_kps)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    best_matches = sorted(matches, key=lambda x: x.distance)[0:400]

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

    h1, w1 = img2.shape[0], img2.shape[1]
    h2, w2 = img1.shape[0], img1.shape[1]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    # plt.plot(pts2.squeeze()[:, 0], pts2.squeeze()[:, 1], "bx")
    # H_pt = []
    # for ptx, pty in pts2.squeeze():
    #     new_pt = np.dot(H, np.array([[ptx], [pty], [1]]))
    #     print("H_pt:", new_pt)
    #     new_pt[0, 0] = new_pt[0, 0] / new_pt[2, 0]
    #     new_pt[1, 0] = new_pt[1, 0] / new_pt[2, 0]
    #     new_pt[2, 0] = new_pt[2, 0] / new_pt[2, 0]
    #     print("H_pt new:", new_pt)
    #     H_pt.append(new_pt)
    # H_pt = np.array(H_pt)
    # print(H_pt)
    # plt.plot(H_pt[:, 0], H_pt[:, 1], "go")
    pts2 = cv2.perspectiveTransform(pts2, H)
    # plt.plot(pts2.squeeze()[:, 0], pts2.squeeze()[:, 1], "rp")
    # plt.show(block=False)
    pts = np.concatenate((pts1, pts2), axis=0)
    print(pts.shape)

    [xmin, ymin] = np.int32(np.min(pts, axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(np.max(pts, axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    transformed_img1 = cv2.warpPerspective(img1, Ht.dot(H), (xmax - xmin, ymax - ymin))
    merged_img = transformed_img1.copy()
    merged_img[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img2

    # finding intersected area of two images
    bw1 = np.zeros(gray1.shape, dtype=np.uint8)
    bw1.fill(1)
    bw2 = np.zeros(gray2.shape, dtype=np.uint8)
    bw2.fill(1)

    merged_bw = cv2.warpPerspective(bw1, Ht.dot(H), (xmax - xmin, ymax - ymin))
    merged_bw[t[1]:h1 + t[1], t[0]:w1 + t[0]] = merged_bw[t[1]:h1 + t[1], t[0]:w1 + t[0]] + bw2

    cropped_img1 = transformed_img1.copy()
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
    added_image = cv2.add(cropped_img1, merged_img)

    cv2.imshow("output1", cleaned_mask)
    cv2.imshow("output2", cropped_img1)
    cv2.imshow("output3", added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

