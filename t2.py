# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

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

    kps = []
    descriptors = []
    output_height = 0
    output_width = 0

    for index, img in enumerate(imgs):
        img = np.array(img)
        output_height += img.shape[0]
        output_width += img.shape[1]
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        kps.append(kp)
        descriptors.append(des)

    for i in range(N):
        for j in range(N):
            if (i == j):
                continue
            matches = bf.match(descriptors[i], descriptors[j])[:1000]
            matches = sorted(matches, key=lambda x: x.distance)
            print(f"matches with kp {i + 1}: ", np.round((len(matches) / len(descriptors[i])) * 100))
            print(f"matches with kp {j + 1}: ", np.round((len(matches) / len(descriptors[j])) * 100))
            # print("matches:", len(matches), (j + 1, i + 1))
            kps1 = np.float32([kp.pt for kp in kps[i]])
            kps2 = np.float32([kp.pt for kp in kps[j]])
            pts1 = np.float32([kps1[m.queryIdx] for m in matches])
            pts2 = np.float32([kps2[m.trainIdx] for m in matches])
            (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            print(np.round(H, decimals=2))
            result = cv2.warpPerspective(imgs[i], H, (imgs[i].shape[1] + imgs[j].shape[1], imgs[i].shape[0] + imgs[j].shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
            # print(np.round(H))
            # result[0:imgs[i].shape[0], 0:imgs[i].shape[1]] = imgs[i]
            # result[0:imgs[j].shape[0], 0:imgs[j].shape[1]] = imgs[j]
            cv2.imshow("test", result)
            cv2.waitKey(0)

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
