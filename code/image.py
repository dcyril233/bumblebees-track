import numpy as np
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv
import cv2
from feature import Feature

class Image:
    """
    store image into a structure
    """
    def __init__(self, input):
        if type(input) is list:
            self.index = input[0]
            self.img = input[1]
            self.record = input[2]
        elif type(input) is dict:
            self.index = input['index']
            self.img = input['img']
            self.record = input['record']

class Data:
    """
    image_path: the path where the data is
    label_path: the path where the label is. If none, user needs to label right now
    version: to know the version of a group in raw data
        0 means (no flash, flash, no flash) is a group
        1 means (flash, no flash) is a group
    del_num: the number of invalid images in the begining
    """
    def __init__(self, image_path, label_path=None, version=1, del_num=0, topn=15):
        self.image_path = image_path
        if version == 0:
            self.img_num = 3
        elif version == 1:
            self.img_num = 2
        self.images = self.import_images(del_num)
        if label_path == None:
            label_path = input("input the path you want to store labels")
            self.label_data(label_path, topn)
            self.label = None
        else:
            self.import_label(label_path)
            self.candidates = self.generate_candidates_by_feature(topn)

    def import_images(self, del_num=0):
        """
        import all the images and delete those images that satisfy the two conditions below:
        1.the extra no-flash image from a group (each group should include two images: flash and no-flash)
        2.the whole group which contains at least an image with none data
        3.the whole group that the brightness of flash image is not high than no-flash image
        """
        # delete data with none
        origin_data = []
        del_pair =[]
        for fn in sorted(glob.glob(self.image_path)):
            raw_data = pickle.load(open(fn,'rb'))
            image = Image(raw_data)
            # delete wrong data
            if image.img is not None:
                image.img = image.img.astype(np.float)
            else:
                del_pair.append(image.index//self.img_num)
            origin_data.append(image)
        if del_num != 0:
            del origin_data[:del_num]
        del_list = []
        for i in range(self.img_num):
            del_list += list(map(lambda x : self.img_num*x+i, del_pair))
        no_none_data = [image for image in origin_data if image.index not in del_list]
        # check if flash works
        del_list = []
        for i in range(0, len(no_none_data)-1, self.img_num):
            if np.mean(no_none_data[i+self.img_num-2].img) <= np.mean(no_none_data[i+self.img_num-1].img) + 0.1:
                for j in range(self.img_num):
                    del_list.append(i+j)
            # delete the extra no-flash imgae
            if self.img_num==3 and i%3==0:
                del_list.append(i)
        data = [no_none_data[i] for i in range(len(no_none_data)) if i not in del_list]
        return data

    def get_dilation(self, f1, nf1, f2, nf2, blocksize=15):
        kernel = np.ones((blocksize, blocksize), np.uint8)
        dilate_nf2 = cv2.dilate(nf2, kernel, iterations = 1)
        dilate_diff_f1_nf1 = cv2.dilate(f1-nf1, kernel, iterations=1)
        img = f2 - dilate_nf2 - dilate_diff_f1_nf1
        return img

    def getmaxlocs(self, img, n=10, deleteradius=20):
        locs = [] # store the coodinates
        for i in range(n):
            cut_img = img[deleteradius:(img.shape[0]-deleteradius), deleteradius:(img.shape[1]-deleteradius)]
            loc = np.unravel_index(cut_img.argmax(), cut_img.shape)
            v = cut_img[loc[0],loc[1]]
            locs.append([loc[0]+deleteradius, loc[1]+deleteradius, v])
            img[(loc[0]):(loc[0]+2*deleteradius),(loc[1]):(loc[1]+2*deleteradius)]=-10000
        return np.array(locs)

    def drawreticule(self, x, y, c='w', alpha=0.5, angle=False):
        """
        draw lines around candidate data
        """
        if angle:
            plt.plot([x-70,x-10],[y-70,y-10],c,alpha=alpha)
            plt.plot([x+70,x+10],[y+70,y+10],c,alpha=alpha)
            plt.plot([x-70,x-10],[y+70,y+10],c,alpha=alpha)
            plt.plot([x+70,x+10],[y-70,y-10],c,alpha=alpha)
        else:
            plt.hlines(y,x-70,x-10,c,alpha=alpha)
            plt.hlines(y,x+10,x+70,c,alpha=alpha)
            plt.vlines(x,y-70,y-10,c,alpha=alpha)
            plt.vlines(x,y+10,y+70,c,alpha=alpha)

    def lowresmaximg(self, img, blocksize=10):
        """
        Downsample image, using maximum from each block
        #from https://stackoverflow.com/questions/18645013/windowed-maximum-in-numpy
        """
        k = int(img.shape[0] / blocksize)
        l = int(img.shape[1] / blocksize)
        if blocksize==1:
            maxes = img
        else:
            maxes = img[:k*blocksize,:l*blocksize].reshape(k,blocksize,l,blocksize).max(axis=(-1,-3))
        return maxes

    def label_data(self, csv_path, topn):
        for i in range(100, len(self.images), 2):
            f1, nf1, f2, nf2 = self.images[i-2].img, self.images[i-1].img, self.images[i].img, self.images[i+1].img
            dilate_img = self.get_dilation(f1, nf1, f2, nf2)
            plt.figure(figsize=[60,20])
            plt.subplot(1, 2, 1)
            plt.imshow(f2)
            plt.clim([0,10])
            plt.subplot(1, 2, 2)
            plt.imshow(f2-nf2)
            plt.clim([0,10])
            # plt.colorbar()
            plt.title([self.images[i].record['triggertimestring'], self.images[i].index])
            print("the %d th group of images" % i)
            locs = self.getmaxlocs(dilate_img, n=topn)
            for j in range(topn):
                y, x = locs[j][0], locs[j][1]
                print(j, x, y)
                self.drawreticule(x, y)
            pos=plt.ginput(n=1, timeout=0)
            print(pos)
            candidate = input('the index of candidate dot is:')
            if candidate != '':
                candidate = int(candidate)
                with open(csv_path, mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    name = 'photo_object_' + self.images[i].record['triggertimestring'].replace(':', '_')
                    index = self.images[i].index
                    name += '_' + "{:04n}".format(index) + '.np'
                    writer.writerow([i, name, locs[candidate][1], locs[candidate][0]])

    def import_label(self, label_path):
        self.label = pd.read_csv(label_path, names=['index','filename', 'x', 'y'])

    def generate_candidates_by_feature(self, topn):
        belonging = []
        x = []
        y = []
        features = []
        labels = []
        for i in range(2, len(self.images), 2):
            f1, nf1, f2, nf2 = self.images[i-2].img, self.images[i-1].img, self.images[i].img, self.images[i+1].img
            locs = self.getmaxlocs_dilation(f1, nf1, f2, nf2, n=topn)
            feature = Feature(locs, f1, nf1, f2, nf2, n=topn).get_feature()
            index = self.images[i].index
            name = 'photo_object_' + self.images[i].record['triggertimestring'].replace(':', '_')
            name += '_' + "{:04n}".format(index) + '.np'
            belonging += [name]*topn
            x += locs[:, 1].tolist()
            y += locs[:, 0].tolist()
            for f in feature:
                features.append(f)
            # store the correct position of true positive data
            dfs = self.label[self.label['filename']==name]
            if len(dfs)>0:
                correctx = dfs['x'].tolist()[0]
                correcty = dfs['y'].tolist()[0]
            else:
                correctx = None
                correcty = None
            for i in range(len(locs)):
                loc = locs[i]
                if loc[1]==correctx and loc[0]==correcty:
                    labels.append(1)
                else:
                    labels.append(0)
        candidates = {'index': np.arange(len(belonging)), 'Name': belonging, 'x':x, 'y':y, 'features':np.array(features), 'labels':np.array(labels)}
        return candidates

    def mark_test_result(self, path, index_list):
        for index in index_list:
            fn = path + self.candidates['Name'][index]
            x = self.candidates['x'][index]
            y = self.candidates['y'][index]
            data = pickle.load(open(fn,'rb'))
            plt.figure(figsize=[25,20])
            img = data['img']
            plt.imshow(img)
            plt.clim([0,10])
            plt.colorbar()
            self.drawreticule(x, y)