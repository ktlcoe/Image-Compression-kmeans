# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:23:35 2021

@author: ktlco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import time


def reshape_image(img):
    shp = img.shape
    r = shp[0]
    c = shp[1]
    z = shp[2]
    new_rows = r*c
    new_cols = z
    return np.reshape(img, (new_rows, new_cols))

def compress_image(img, k):
    start_time = time.time()
    shape = img.shape
    rows = shape[0]
    cols = shape[1]
    # step 1: choose initial clusters
    new_c = img[np.random.randint(0, rows, k), :]
    c = new_c+1
    
    # this is used to answer question 2 part b - what if we intentionally choose bad initial centroids
    # bad_c
    # new_c = np.array([[0,0,0], [255, 255, 255]])
    # new_c = np.array([[0,0,0], [1,1,1], [254, 254, 254], [255, 255, 255]])
    # new_c = np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3], [252,252,252], [253,253,253], [254, 254, 254], [255, 255, 255]])
    # new_c = np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7], [248, 248, 248], [249, 249, 249], [250, 250, 250], [251,251,251], [252,252,252], [253,253,253], [254, 254, 254], [255, 255, 255]])
    
    
    
    # set up iteration loop
    iter = 1
    while not np.array_equal(c, new_c):
        print("--iteration %d" % iter)
        c = new_c
        # calculate distances from clusters
        diffs = np.array([], dtype=np.float32).reshape(rows, 0)
        for i in range(0, k):
            tmp = np.sqrt(np.sum(np.square(img - c[i]), axis=1)).reshape(rows, 1)
            diffs = np.concatenate([diffs, tmp], axis=1)
            
        # assign each point to smallest distance cluster
        l = np.argmin(diffs, axis=1)
        labels = l.reshape(rows, 1)        
        
        # create data assignment matrix - taken from demo code
        P = csc_matrix((np.ones(rows), (np.arange(0, rows, 1), l)), shape=(rows, k))
        count = P.sum(axis=0)
                
        # recalculate centroids
        # formula taken from demo code
        new_c = np.array(P.T.dot(img).T / count).T
        iter+=1
        
    print("--- %.3f seconds ---" % (time.time() - start_time))
    return (labels, new_c)

def display_image(labels, centroids, original_r, original_c):
    shp = labels.shape
    rows = shp[0]
    idx = labels.reshape(rows,)
    img = centroids[idx].reshape(original_r, original_c, 3)
    plt.imshow(img.astype(np.uint8))
    return img.astype(np.uint8)

test_k = [2,4,8,16]

beach_image = plt.imread("data/beach.bmp").astype(np.float32)
original_r = beach_image.shape[0]
original_c = beach_image.shape[1]
beach = reshape_image(beach_image)
for k in test_k:
    (labels, centroids) = compress_image(beach, k)
    new_img = display_image(labels, centroids, original_r, original_c)
    plt.imsave("data/beach_" + str(k) + ".png", new_img)

# football_image = plt.imread("data/football.bmp").astype(np.float32)
# original_r = football_image.shape[0]
# original_c = football_image.shape[1]
# football = reshape_image(football_image)
# for k in test_k:
#     (labels, centroids) = compress_image(football, k)    
#     new_img = display_image(labels, centroids, original_r, original_c)
#     plt.imsave("data/football_" + str(k) + ".png", new_img)

# turtle_image = plt.imread("data/turtle.bmp").astype(np.float32)
# original_r = turtle_image.shape[0]
# original_c = turtle_image.shape[1]
# turtle = reshape_image(turtle_image)
# for k in test_k:
#     (labels, centroids) = compress_image(turtle, k)    
#     new_img = display_image(labels, centroids, original_r, original_c)
#     plt.imsave("data/turtle_" + str(k) + ".png", new_img)
    
# # used to test intentionally bad centroids with beach image
# beach_image = plt.imread("data/beach.bmp").astype(np.float32)
# original_r = beach_image.shape[0]
# original_c = beach_image.shape[1]
# beach = reshape_image(beach_image)
# k=16
# (labels, centroids) = compress_image(beach, k)
# new_img = display_image(labels, centroids, original_r, original_c)
# plt.imsave("data/beach_bad_" + str(k) + ".png", new_img)









