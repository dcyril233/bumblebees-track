import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
from feature import Feature, ShapeFactor
from image import Data
from model import LeaveOneOutModel, TrainValTestModel
from preprocess import Preprocess
from visualize import Visualize
import cv2

np.random.seed(42)

binary_mask = np.array([[0., 1., 1., 1., 0., 0., 0., 0., 0.],
                        [0., 1., 1., 1., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 0., 1., 0., 0., 0.],
                        [1., 1., 1., 1., 1., 1., 0., 0., 0.],
                        [1., 1., 1., 0., 1., 0., 0., 0., 0.],
                        [1., 1., 0., 0., 1., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0.]])

keep = np.array([[4, 4],[5, 4],[6, 4],[4, 5],[3, 5],[4, 3],[3, 3],[2, 3],[1, 3],[0, 3],[0, 2],[1, 2],[2, 2],[3, 2],[4, 2],[5, 2],[5, 1],[6, 1],[6, 0],[7, 0],[5, 0],[4, 0],[3, 0],[2, 0],[2, 1],[3, 1],[4, 1],[1, 1],[0, 1]])

factor = ShapeFactor(test=True)
area = factor.get_area(keep)
perimeter = factor.get_perimeter(keep, binary_mask)
print(area)
print(perimeter)


plt.figure()
plt.title('binary_mask')
plt.imshow(binary_mask, vmin=0, vmax=1)
plt.show()
"""
f1 = pickle.load(open('C:/dissertation/dataset/photos_April4_location1/photo_object_20200404_11_36_16.451732_0004.np','rb'))[1].astype(np.float)
patch_size = 7
r, c = 991, 1031
r1, r2 = int(r-patch_size//2), int(r+patch_size//2)+1
c1, c2 = int(c-patch_size//2), int(c+patch_size//2)+1
patch = f1[r1:r2, c1:c2]

feature = Feature(test=True)
edge_patch = feature.detect_edge([patch])[0][0]
edge_patch2 = cv2.Laplacian(patch, cv2.CV_64F)

plt.figure()
plt.subplot(1, 2, 1)
plt.title('original patch')
plt.imshow(patch, vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.title('edge detection with laplace')
plt.imshow(edge_patch2, vmin=0, vmax=255)
plt.show()


a = np.array([2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0])
print(np.mean(a))
print(np.std(a))

img = np.random.randint(3, size=(7, 7)).astype(np.int16)
print(img)
feature = Feature(test=True)
mean_ring, mean_std = feature.get_concentric_ring(locs=[[3, 3]], img=img, r=2)
print(mean_ring, mean_std)

patch_f2 = np.array([[0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1],
                     [0, 1, 0, 1, 1]]).astype(np.int16)

patch_nf2 = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0]]).astype(np.int16)

patch_f1 = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1],
                     [0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0]]).astype(np.int16)

patch_nf1 = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]).astype(np.int16)

blocksize = 3
diff2 = patch_f2 - patch_nf2
kernel = np.ones((blocksize, blocksize), np.uint8)
dilate_nf2 = cv2.dilate(patch_nf2, kernel, iterations=1)
dilate_diff2 = patch_f2 - dilate_nf2
diff1 = patch_f1 - patch_nf1
dilate_diff1 = cv2.dilate(diff1, kernel, iterations=1)

test = Data(test=True)

dilated_img = test.get_dilation(patch_f1, patch_nf1, patch_f2, patch_nf2, blocksize=3)
plt.figure()
plt.subplot(2, 5, 1)
plt.title('previous flash')
plt.imshow(patch_f1, vmin=0, vmax=1)
plt.subplot(2, 5, 2)
plt.title('previous noflash')
plt.imshow(patch_nf1, vmin=0, vmax=1)
plt.subplot(2, 5, 3)
plt.title('previous diff')
plt.imshow(patch_f1-patch_nf1, vmin=0, vmax=1)
plt.subplot(2, 5, 4)
plt.title('previous dilated diff')
plt.imshow(dilate_diff1, vmin=0, vmax=1)
plt.subplot(2, 5, 5)
plt.title('supposed result')
plt.imshow(dilate_diff2-dilate_diff1, vmin=0, vmax=1)
plt.subplot(2, 5, 6)
plt.title('current flash')
plt.imshow(patch_f2, vmin=0, vmax=1)
plt.subplot(2, 5, 7)
plt.title('current noflash')
plt.imshow(patch_nf2, vmin=0, vmax=1)
plt.subplot(2, 5, 8)
plt.title('current diff')
plt.imshow(patch_f2-patch_nf2, vmin=0, vmax=1)
plt.subplot(2, 5, 9)
plt.title('current flash-dilated(noflash)')
plt.imshow(dilate_diff2, vmin=0, vmax=1)
plt.subplot(2, 5, 10)
plt.title('output result')
plt.imshow(dilated_img, vmin=0, vmax=1)
plt.show()

test_array = np.random.randint(255, size=(5, 5))
print(test_array)
locs = test.getmaxlocs(test_array, n=2, deleteradius=1)
print(locs)
print(test_array)
"""