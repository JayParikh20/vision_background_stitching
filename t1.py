# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    gray1_kps = cv2.drawKeypoints(gray1, kp1, gray1)
    cv2.imwrite('img1_kps.jpg', gray1_kps)

    kp2, des2 = sift.detectAndCompute(gray2, None)
    gray2_kps = cv2.drawKeypoints(gray2, kp2, gray2)
    cv2.imwrite('img2_kps.jpg', gray2_kps)

    print("des1 type:", type(des1))
    print("des2 type:", type(des2))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3), plt.show()

    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    print('M', M)
    # h, w = np.array(gray1).shape
    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    # gray2 = cv2.polylines(gray2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    #
    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matches_mask,  # draw only inliers
    #                    flags=2)
    # img3 = cv2.drawMatches(gray1, kp1, gray2, kp2, good, None, **draw_params)
    # plt.imshow(img3, 'gray'), plt.show()

    width = gray1.shape[1] + gray2.shape[1]
    height = gray1.shape[0] + gray2.shape[0]

    result = cv2.warpPerspective(gray1, M, (749, 400))
    result[0:gray2.shape[0], 0:gray2.shape[1]] = gray2
    plt.imshow(result), plt.show()
    cv2.imwrite("output.jpg", result)

    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

