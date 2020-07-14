import cv2
import numpy as np
import skimage
import sklearn
from skimage.color import label2rgb
from skimage.util import random_noise
from sklearn.cluster import estimate_bandwidth, MeanShift, MiniBatchKMeans
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
import os
import pandas as pd

#########################################
# SimonaMnv								#
#########################################


def img_print(blabla, img):
    cv2.namedWindow(blabla, cv2.WINDOW_NORMAL)
    cv2.imshow(blabla, img)
    # cv2.imwrite('resultimage0.jpg', img/255)
    cv2.resizeWindow(blabla, 450, 450)
    cv2.waitKey(0)


def color_hist_similarity(img_patch, target):
    # Custom func that calculates the similarity based on color histogram
    template_hist = cv2.calcHist([target], [0], None, [256], [0, 256])
    patch_hist = cv2.calcHist([img_patch], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(template_hist, patch_hist, 0)  # img_patch histogram each time
    return hist_score


def template_matching(img, target, thres):
    height, width = img.shape[:2]
    tar_height, tar_width = target.shape[:2]
    NccValue = np.zeros((height - tar_height, width - tar_width))
    loc = []
    mask = np.ones((tar_height, tar_width, 3))
    maxScores = []

    # Create a sliding template-window
    for m in range(0, height - tar_height):
        for n in range(0, width - tar_width):
            img_patch = img[m: m + tar_height, n: n + tar_width]  # scan through the moving window
            # img_print("Sliding Window", skimage.util.img_as_ubyte(img))  # check if the window moves
            NccValue[m, n] = color_hist_similarity(img_patch, target)  # calculate histogram of each image patch
            if NccValue[m, n] > thres:  # check if that patch is above the threshold
                maxScores.append(NccValue[m, n])
                offset = np.array((m, n))  # use the mask to spread out the box detection
                img[offset[0]:offset[0] + mask.shape[0], offset[1]:offset[1] + mask.shape[1]] = mask
                (max_Y, max_X) = (m, n)
                loc.append((max_X, max_Y))  # appends backwards!
    return loc, maxScores


def question2():
    image = cv2.imread('images to use/9.JPG', cv2.IMREAD_COLOR)
    image2 = cv2.imread('images to use/9.jpg',
                        cv2.IMREAD_COLOR)  # the original has the mask above, so print a clear one
    target = cv2.imread('images to use/9-template.jpg', cv2.IMREAD_COLOR)
    height, width = target.shape[:2]

    top_left, scores = template_matching(image, target, 0.80)  # Set the threshold here!!!

    for i in range(0, len(top_left)):  # Print all the boxes found in the threshold range
        loc = top_left[i]
        cv2.rectangle(image2, loc, (loc[0] + width, loc[1] + height), (0, 0, 255), 3)
        cv2.putText(image2, str(round(scores[i], 4)), (loc[0] + width - 100, loc[1] + height), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    img_print("Object(s) detected", image2)


question1()