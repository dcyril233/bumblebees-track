import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
from feature import Feature
from image import Image
from model import Model
from preprocess import Preprocess
from visualize import Visualize

# get labels of whole data based on indexes of pos data 
def get_label(ls, length):
    label = [ 1 if x in ls else 0 for x in range(length) ]
    return np.array(label)


if __name__ == '__main__':

    # import data
    data = Image('D:/U/dissertation/dataset/photos_April4/photo_object*').import_data(del_num=6)
    
    index_pos_april4 = [8, 13, 23, 28, 33, 38, 42, 63, 68, 74, 83, 88, 93, 96, 108, 113, 128, 133, 143, 148, 153, 157, 168, 172, 182, 193, 197, 206, 299, 333, 342, 382, 401, 474]
    # start from second pair
    index_pos_april4 = [i - 5 for i in index_pos_april4]

    index_pos_april5 = [15, 22, 27, 32, 37, 43, 54, 58, 64, 69, 73, 89, 93, 98, 108, 119, 123, 133, 148, 152, 172, 177, 182, 187, 193, 209, 268, 277, 282, 287, 292, 300, 306, 310, 316, 320, 327, 331, 337, 353, 359, 411, 416, 422, 428, 433, 463, 513, 517, 565, 585, 590, 595, 674, 678, 688, 698, 710, 719, 745, 750, 755, 760, 765, 770, 775, 780, 785, 790, 795, 804, 810, 815]
    # start from second pair
    index_pos_april5 = [i - 5 for i in index_pos_april5]

    index_pos_april13 = [43, 48, 52, 63, 68, 78, 83, 88, 98, 118, 128, 133, 138, 143, 148, 153, 163, 168, 172, 194, 198, 203, 208, 213, 243, 249, 263, 279, 281]
    # start from second pair
    index_pos_april13 = [i - 5 for i in index_pos_april13]

    # extract features and labels
    features = np.empty((0, 624))
    for i in range(3, len(data)-1,3):
        feature = Feature(data[i][1], data[i+2][1], data[i+1][1], data[i-2][1])
        output = feature.get_feature()
        features = np.append(features, output, axis=0)

    index_feature = feature.get_index()

    label = get_label(index_pos_april4, features.shape[0])
    
    lr = Model(features, label, 'LR')
    # lr.test_false_positive()
    # lr.get_metrics(0.25)
    importance = lr.get_important_feature()
    imp_index_pos = importance.argsort()[::-1]
    imp_index_neg = importance.argsort()
    feature.get_belong_feature(imp_index_pos[:10])

    visual = Visualize(importance, imp_index_pos)
    visual.plot_feature(len(imp_index_pos))