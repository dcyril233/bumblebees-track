import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
from feature import Feature
from image import Data
from model import LeaveOneOutModel
from preprocess import Preprocess
from visualize import Visualize

if __name__ == '__main__':

    image_path_july8 = 'C:/dissertation/dataset/photos20200608/photo_object*'
    # 'C:/dissertation/dataset/labels_photos_July08.csv'
    data_july8 = Data(image_path=image_path_july8, label_path=None, version=1, del_num=8, topn=5)

    """
    image_path_april4 = 'C:/dissertation/dataset/photos_April4/photo_object*'
    # 'labels_photos_April05.csv'
    data_april4 = Data(image_path=image_path_april4, label_path=None, version=0, del_num=6, topn=20)

    image_path_april5 = 'C:/dissertation/dataset/photos_April5/photo_object*'
    # 'labels_photos_April05.csv'
    data_april5 = Data(image_path=image_path_april5, label_path=None, version=0, del_num=0, topn=20)

    image_path_april7 = 'C:/dissertation/dataset/photos_April7/photo_object*'
    data_april7 = Data(image_path=image_path_april7, label_path=None, version=0, del_num=0, topn=20)

    image_path_april13 = 'C:/dissertation/dataset/photos_April13/photo_object*'
    data_april13 = Data(image_path=image_path_april13, label_path=None, version=0, del_num=6, topn=20)

    image_path_july8 = 'C:/dissertation/dataset/photos20200608/photo_object*'
    # 'C:/dissertation/dataset/labels_photos_July08.csv'
    data_july8 = Data(image_path=image_path_july8, label_path=None, version=1, del_num=8, topn=20)

    """
    
    """
    # import data
    image_path = 'C:/dissertation/dataset/photos_April4/photo_object*'
    label_path = 'C:/dissertation/dataset/labels_photos_April04.csv'
    candidates_april5 = Data(image_path=image_path, label_path=label_path, version=0, del_num=6).generate_candidates(topn=20, boxsize=4, threshold=120)

    image_path = 'C:/dissertation/dataset/photos20200608/photo_object*'
    label_path = 'C:/dissertation/dataset/labels_photos_July08.csv'
    candidates_july8 = Data(image_path=image_path, label_path=label_path, version=1, del_num=8).generate_candidates(topn=20, boxsize=4, threshold=120)
    X_train = candidates_april5['features']
    X_test = candidates_july8['features']
    Y_train = candidates_april5['labels']
    Y_test = candidates_july8['labels']
    leave_one_cv = LeaveOneOutModel(X_train, X_test, Y_train, Y_test, 'RF')
    visual = Visualize()
    visual.plot_roc_curve(leave_one_cv.fpr, leave_one_cv.tpr)

    bst_threshold=0.5
    visual.plot_confusion(leave_one_cv.clf, leave_one_cv.X_train, leave_one_cv.Y_train, bst_threshold)
    visual.plot_confusion(leave_one_cv.clf, leave_one_cv.X_test, leave_one_cv.Y_test, bst_threshold)

    false_positives_list = leave_one_cv.false_positive_index(leave_one_cv.clf, leave_one_cv.X_test, leave_one_cv.Y_test, bst_threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = candidates_april5['factors'][:, 0]
    ys = candidates_april5['factors'][:, 1]
    zs = Y_train
    ax.scatter(xs, ys, zs, c=zs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = candidates_july8['factors'][:, 0]
    ys = candidates_july8['factors'][:, 1]
    zs = Y_test
    ax.scatter(xs, ys, zs, c=zs)
    """