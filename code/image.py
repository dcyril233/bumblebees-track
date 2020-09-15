import numpy as np
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv
import cv2
from feature import Feature, ShapeFactor

class Image:
    """
    store image into a structure
    """
    def __init__(self, name_list, index):
        fn = name_list[index]
        input_data = pickle.load(open(fn, 'rb'))
        if type(input_data) is list:
            self.index = input_data[0]
            self.img = input_data[1]
            self.record = input_data[2]
        elif type(input_data) is dict:
            self.index = input_data['index']
            self.img = input_data['img']
            self.record = input_data['record'] 

class Data:
    """
    image_path: the path where the data is
    label_path: the path where the label is. If none, user needs to label right now
    version: to know the version of a group in raw data
        0 means (no flash, flash, no flash) is a group
        1 means (flash, no flash) is a group
    del_num: the number of invalid images in the begining
    """
    def __init__(self, image_path=None, label_path=None, version=1, del_num=0, topn=15, relabel=False, test=False, ignore_area=None):
        if test == False:
            self.image_path = image_path
            self.relabel = relabel
            if version == 0:
                self.img_num = 3
            elif version == 1:
                self.img_num = 2
            self.images = self.import_images(del_num, ignore_area)
            if label_path == None:
                label_path = input("input the path you want to store labels:")
                self.label_data(label_path, topn)
                self.label = None
            else:
                self.import_label(label_path)
                if relabel == True:
                    self.label_data(label_path, topn)

    def import_images(self, del_num=0, ignore_area=None):
        """
        import all the images and delete those images that satisfy the two conditions below:
        1.the extra no-flash image from a group (each group should include two images: flash and no-flash)
        2.the whole group which contains at least an image with none data
        3.the whole group that the brightness of flash image is not high than no-flash image
        """
        # delete data with none
        data =[]
        name_list = sorted(glob.glob(self.image_path))
        if del_num != 0:
            del name_list[:del_num]
        # loop all the images
        for i in range(0, len(name_list), self.img_num):
            group = []
            flash_index = i+self.img_num-2
            no_flash_index = list(set(range(i, i+self.img_num))- set([flash_index]))
            # check the flash image
            flash_data = Image(name_list, flash_index)
            if flash_data.img is None:
                continue
            else:
                flash_bright= np.mean(flash_data.img)
            # check the no-flash image
            for index in no_flash_index:
                no_flash_data = Image(name_list, index)
                if no_flash_data.img is not None:
                    no_flash_bright = np.mean(no_flash_data.img)
                    if flash_bright > no_flash_bright + 0.1:
                        group = [flash_data, no_flash_data]
                        break
            if group == []:
                continue
            for image in group:
                # normalize all the img
                # img = image.img.astype(np.float32)
                # cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                # image.img = img.astype(np.int16)
                image.img = image.img.astype(np.int16)
                if ignore_area is not None:
                    # cut out the area including specially false dots in data on April 13
                    image.img[:ignore_area[0], :ignore_area[1]] = -10000
                data.append(image)
        return data

    def get_dilation(self, f1, nf1, f2, nf2, blocksize=15):
        copy_f1, copy_nf1, copy_f2, copy_nf2 = f1.copy(), nf1.copy(), f2.copy(), nf2.copy()
        kernel = np.ones((blocksize, blocksize), np.uint8)
        dilate_nf2 = cv2.dilate(copy_nf2, kernel, iterations=1)
        dilate_diff_f1_nf1 = cv2.dilate(copy_f1-copy_nf1, kernel, iterations=1)
        img = copy_f2 - dilate_nf2 - dilate_diff_f1_nf1
        return img

    def getmaxlocs(self, img, n=10, deleteradius=40):
        copy_img = img.copy()
        locs = [] # store the coodinates
        for i in range(n):
            cut_img = copy_img[deleteradius:(copy_img.shape[0]-deleteradius), deleteradius:(copy_img.shape[1]-deleteradius)]
            loc = np.unravel_index(cut_img.argmax(), cut_img.shape)
            v = cut_img[loc[0],loc[1]]
            locs.append([loc[0]+deleteradius, loc[1]+deleteradius, v])
            copy_img[(loc[0]):(loc[0]+2*deleteradius),(loc[1]):(loc[1]+2*deleteradius)]=-10000
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

    def get_intersection_rows(self, dilate_img, diff_img, topn):
        dilate_locs = self.getmaxlocs(dilate_img, n=topn)
        diff_locs = self.getmaxlocs(diff_img, n=topn)
        df1 = pd.DataFrame({'row': dilate_locs[:, 0], 
                           'column': dilate_locs[:, 1],
                           'value': dilate_locs[:, 2]})
        df2 = pd.DataFrame({'row': diff_locs[:, 0], 
                           'column': diff_locs[:, 1],
                           'value': diff_locs[:, 2]})
        interpd = df1.merge(df2, on=['row', 'column'], how='inner')
        interpd = interpd.drop('value_x', 1)
        # make sure the brightest dot in dilation image be in the candidates
        brightest = df1.iloc[0]
        if not ((interpd['row'] == brightest['row']) & (interpd['column'] == brightest['column'])).any():
            r = int(brightest['row'])
            c = int(brightest['column'])
            v = diff_img[r, c]
            interpd.loc[interpd.shape[0]] = [r, c, v]
        return interpd.values, dilate_locs, diff_locs

    def label_data(self, csv_path, topn):
        for i in range(2, len(self.images)-2, 2):
            f1, nf1, f2, nf2, f3, nf3 = self.images[i-2].img, self.images[i-1].img, self.images[i].img, self.images[i+1].img, self.images[i+2].img, self.images[i+3].img
            if self.relabel == True and self.images[i].index in self.label['index'].values:
                continue
            past_dilate_img = self.get_dilation(f1, nf1, f2, nf2)
            future_dilate_img = self.get_dilation(f3, nf3, f2, nf2)
            locs, past_dilate_locs, future_dilate_locs = self.get_intersection_rows(past_dilate_img, future_dilate_img, topn)
            diff_img = f2 - nf2
            plt.figure(figsize=[60,20])
            plt.subplot(2, 2, 1)
            plt.imshow(f2)
            plt.title('flash image')
            # plt.clim([0,10])
            plt.subplot(2, 2, 2)
            plt.imshow(past_dilate_img, vmin=0, vmax=255)
            plt.title('the seond difference between the current and the previous')
            # plt.clim([0,10])
            for j in range(len(past_dilate_locs)):
                y, x = past_dilate_locs[j][0], past_dilate_locs[j][1]
                self.drawreticule(x, y)
            # plt.colorbar()
            plt.subplot(2, 2, 3)
            plt.imshow(future_dilate_img, vmin=0, vmax=255)
            plt.title('the seond difference between the current and the next')
            # plt.clim([0,10])
            for j in range(len(future_dilate_locs)):
                y, x = future_dilate_locs[j][0], future_dilate_locs[j][1]
                self.drawreticule(x, y)
            plt.subplot(2, 2, 4)
            plt.imshow(f2)
            # plt.clim([0,10])
            plt.title([self.images[i].record['triggertimestring'], self.images[i].index])
            print("the %d th group of images" % i)
            for j in range(len(locs)):
                y, x = locs[j][0], locs[j][1]
                print(j, x, y)
                self.drawreticule(x, y)
            pos = plt.ginput(n=1, timeout=0)
            print(pos)
            # candidate = input('the index of candidate dot is:')
            # if candidate != '':
            #    candidate = int(candidate)
            candidate = -1
            mindistance = 50
            for j in range(len(locs)):
                y, x = locs[j][0], locs[j][1]
                if (pos[0][0] - x)*(pos[0][0] - x) + (pos[0][1] - y)*(pos[0][1] - y) < mindistance:
                    candidate = j
                    mindistance = (pos[0][0] - x)*(pos[0][0] - x) + (pos[0][1] - y)*(pos[0][1] - y)
            print('candidate = %d, distance = %d ' % (candidate, mindistance))
            if candidate != -1:
                with open(csv_path, mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    name = 'photo_object_' + self.images[i].record['triggertimestring'].replace(':', '_')
                    index = self.images[i].index
                    name += '_' + "{:04n}".format(index) + '.np'
                    writer.writerow([index, name, locs[candidate][1], locs[candidate][0]])
            plt.close()

    def import_label(self, label_path):
        self.label = pd.read_csv(label_path, names=['index','filename', 'x', 'y'])

    def get_neighbour_imple(self, patch, point, ban, keep):
        patch_size = len(patch)
        if point[0] < patch_size-1:
            row = int(point[0] + 1)
            col = int(point[1])
            value = patch[row, col]
            next_point = [row, col]
            if next_point not in ban and next_point not in keep and value == 1:
                keep.append(next_point)
                self.get_neighbour_imple(patch, next_point, ban, keep)
            elif next_point not in ban:
                ban.append(next_point)
        if point[0] > 0:
            row = int(point[0] - 1)
            col = int(point[1])
            value = patch[row, col]
            next_point = [row, col]
            if next_point not in ban and next_point not in keep and value == 1:
                keep.append(next_point)
                self.get_neighbour_imple(patch, next_point, ban, keep)
            elif next_point not in ban:
                ban.append(next_point)
        if point[1] < patch_size-1:
            row = int(point[0])
            col = int(point[1] + 1)
            value = patch[row, col]
            next_point = [row, col]
            if next_point not in ban and next_point not in keep and value == 1:
                keep.append(next_point)
                self.get_neighbour_imple(patch, next_point, ban, keep)
            elif next_point not in ban:
                ban.append(next_point)
        if point[1] > 0:
            row = int(point[0])
            col = int(point[1] - 1)
            value = patch[row, col]
            next_point = [row, col]
            if next_point not in ban and next_point not in keep and value == 1:
                keep.append(next_point)
                self.get_neighbour_imple(patch, next_point, ban, keep)
            elif next_point not in ban:
                ban.append(next_point)

    def get_neighbour(self, patch, point):
        keep = []
        keep.append(point)
        ban = []
        self.get_neighbour_imple(patch, point, ban, keep)
        binary_mask = np.zeros(patch.shape)
        for loc in keep:
            row = int(loc[0])
            col = int(loc[1])
            binary_mask[row, col] = 1
        return binary_mask, np.array(keep)

    def get_mean_diff(self):
        diff_img = np.zeros((1536, 2048))
        for i in range(0, len(self.images), 2):
            f2, nf2 = self.images[i].img, self.images[i+1].img
            diff = f2 - nf2
            diff_img += diff
            pixel_values = diff.flatten()
            plt.hist(pixel_values, 40, facecolor='blue', alpha=0.5, log=True)
            plt.show()
        return diff_img/len(self.images)

    def get_factors(self, locs, diff_img, boxsize, threshold):
        factor = []
        for loc in locs:
            binary_img = np.where(diff_img > threshold, 1, 0)
            r, c = loc[0], loc[1]
            r1, r2 = int(r-boxsize), int(r+boxsize)+1
            c1, c2 = int(c-boxsize), int(c+boxsize)+1
            binary_patch = binary_img[r1:r2, c1:c2]
            point = [boxsize, boxsize]
            binary_mask, keep = self.get_neighbour(binary_patch, point)
            f = ShapeFactor(binary_mask, keep).get_factors()
            factor.append(f)
        return factor
            
    def generate_candidates(self, topn, boxsize, threshold, blocksize=41):
        belonging = []
        x = []
        y = []
        features = []
        factors = []
        labels = []
        for i in range(2, len(self.images)-2, 2):
            f1, nf1, f2, nf2, f3, nf3 = self.images[i-2].img, self.images[i-1].img, self.images[i].img, self.images[i+1].img, self.images[i+2].img, self.images[i+3].img
            past_dilate_img = self.get_dilation(f1, nf1, f2, nf2, blocksize)
            future_dilate_img = self.get_dilation(f3, nf3, f2, nf2, blocksize)
            locs, past_dilate_locs, future_dilate_locs = self.get_intersection_rows(past_dilate_img, future_dilate_img, topn)
            diff_img = f2 - nf2
            feature = Feature(locs, f1, nf1, f2, nf2, f3, nf3, past_dilate_img, future_dilate_img).get_feature()
            for f in feature:
                features.append(f)
            factor = self.get_factors(locs, diff_img, boxsize, threshold)
            factors += factor
            index = self.images[i].index
            name = 'photo_object_' + self.images[i].record['triggertimestring'].replace(':', '_')
            name += '_' + "{:04n}".format(index) + '.np'
            belonging += [name]*len(locs)
            x += locs[:, 1].tolist()
            y += locs[:, 0].tolist()
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
        candidates = {'index': np.arange(len(belonging)), 'Name': belonging, 'x':x, 'y':y, 'features':np.array(features), 'factors':np.array(factors), 'labels':np.array(labels)}
        return candidates
           
    def mark_test_result(self, candidates, path, index_list):
        for index in index_list:
            fn = path + candidates['Name'][index]
            x = candidates['x'][index]
            y = candidates['y'][index]
            data = pickle.load(open(fn,'rb'))
            plt.figure(figsize=[25,20])
            img = data['img']
            plt.imshow(img)
            plt.clim([0,10])
            plt.colorbar()
            self.drawreticule(x, y)