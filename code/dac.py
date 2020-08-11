import numpy as np
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv
import cv2
from feature import Feature

def get_dilation(f1, nf1, f2, nf2, blocksize=15):
    kernel = np.ones((blocksize, blocksize), np.uint8)
    dilate_nf2 = cv2.dilate(nf2, kernel, iterations = 1)
    dilate_diff_f1_nf1 = cv2.dilate(f1-nf1, kernel, iterations=1)
    img = f2 - dilate_nf2 - dilate_diff_f1_nf1
    return img

def getmaxlocs(img, n=10, deleteradius=20):
    copy_img = img.copy()
    locs = [] # store the coodinates
    for i in range(n):
        cut_img = copy_img[deleteradius:(copy_img.shape[0]-deleteradius), deleteradius:(copy_img.shape[1]-deleteradius)]
        loc = np.unravel_index(cut_img.argmax(), cut_img.shape)
        v = cut_img[loc[0],loc[1]]
        locs.append([loc[0]+deleteradius, loc[1]+deleteradius, v])
        copy_img[(loc[0]):(loc[0]+2*deleteradius),(loc[1]):(loc[1]+2*deleteradius)]=-10000
    return np.array(locs)


def get_neighbour_imple(patch, point, ban, keep, max_value, epsilon):
    patch_size = len(patch)
    if point[0] < patch_size-1:
        row = int(point[0] + 1)
        col = int(point[1])
        value = patch[row, col]
        next_point = [row, col, value]
        differance = abs(max_value-next_point[2])
        if next_point not in ban and next_point not in keep and differance < epsilon:
            keep.append(next_point)
            get_neighbour_imple(patch, next_point, ban, keep, max_value, epsilon)
        elif next_point not in ban:
            ban.append(next_point)
    if point[0] > 0:
        row = int(point[0] - 1)
        col = int(point[1])
        value = patch[row, col]
        next_point = [row, col, value]
        differance = abs(max_value-next_point[2])
        if next_point not in ban and next_point not in keep and differance < epsilon:
            keep.append(next_point)
            get_neighbour_imple(patch, next_point, ban, keep, max_value, epsilon)
        elif next_point not in ban:
            ban.append(next_point)
    if point[1] < patch_size-1:
        row = int(point[0])
        col = int(point[1] + 1)
        value = patch[row, col]
        next_point = [row, col, value]
        differance = abs(max_value-next_point[2])
        if next_point not in ban and next_point not in keep and differance < epsilon:
            keep.append(next_point)
            get_neighbour_imple(patch, next_point, ban, keep, max_value, epsilon)
        elif next_point not in ban:
            ban.append(next_point)
    if point[1] > 0:
        row = int(point[0])
        col = int(point[1] - 1)
        value = patch[row, col]
        next_point = [row, col, value]
        differance = abs(max_value-next_point[2])
        if next_point not in ban and next_point not in keep and differance < epsilon:
            keep.append(next_point)
            get_neighbour_imple(patch, next_point, ban, keep, max_value, epsilon)
        elif next_point not in ban:
            ban.append(next_point)


def get_neighbour(patch, point, epsilon):
    keep = []
    keep.append(point)
    ban = []
    max_value = point[2]
    get_neighbour_imple(patch, point, ban, keep, max_value, epsilon)
    binary_mask = np.zeros(patch.shape)
    for loc in keep:
        row = int(loc[0])
        col = int(loc[1])
        binary_mask[row, col] = 1
    return binary_mask


f2 = pickle.load(open('C:/dissertation/dataset/photos_April4/photo_object_20200404_11_58_40.336292_0274.np','rb'))[1].astype(np.float)
nf2 = pickle.load(open('C:/dissertation/dataset/photos_April4/photo_object_20200404_11_58_40.380200_0275.np','rb'))[1].astype(np.float)

diff_image = f2 - nf2

c, r = 1234, 663
r1, r2 = int(r-4), int(r+4)+1
c1, c2 = int(c-4), int(c+4)+1

print(diff_image[r1:r2, c1:c2])