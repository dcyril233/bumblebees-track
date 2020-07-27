import numpy as np
import glob
import pickle
import pandas as pd


class Image:
    
    
    # create object storing path of data
    def __init__(self, image_path, label_path, version):
        self.image_path = image_path
        self.label_path = label_path
        self.version = version
        if version == 0:
            self.img_num = 3
        elif version == 1:
            self.img_num = 2


    # import data
    def import_data(self, del_num=0):
        # load label
        df = pd.read_csv(self.label_path, names=['index','filename', 'x', 'y'])
        # delete data with none
        origin_data = []
        del_pair =[]
        for fn in sorted(glob.glob(self.image_path)):
            raw_data = pickle.load(open(fn,'rb'))
            # convert the type of data from list to dict
            if self.version == 0:
                dat = {}
                dat['index'] = raw_data[0]
                dat['img'] = raw_data[1]
                dat['record'] = raw_data[2]
            elif self.version == 1:
                dat = raw_data
            # store the correct position of true positive data
            dfs = df[df['filename']==fn.split('/')[-1].split('\\')[-1]]
            # add the coordinate of true positive dot
            correctx = None
            correcty = None
            if len(dfs)>0: 
                correctx = dfs['x'].tolist()[0]
                correcty = dfs['y'].tolist()[0]
            dat['correctx'] = correctx
            dat['correcty'] = correcty
            # delete wrong data
            if dat['img'] is not None:
                dat['img'] = dat['img'].astype(np.float)
            else:
                del_pair.append(dat['index']//self.img_num)
            origin_data.append(dat)
        if del_num != 0:
            del origin_data[:del_num]

        # del_list = list(map(lambda x : 3*x, del_pair)) + list(map(lambda x : 3*x + 1, del_pair)) + list (map(lambda x : 3*x + 2, del_pair))
        del_list = []
        for i in range(self.img_num):
            del_list += list(map(lambda x : self.img_num*x+i, del_pair))
        no_none_data = [image for image in origin_data if image['index'] not in del_list]

        # check if flash works
        del_list = []
        # for i in range(0, len(no_none_data)-1, 3):
        for i in range(0, len(no_none_data)-1, self.img_num):
            if np.mean(no_none_data[i+self.img_num-2]['img']) <= np.mean(no_none_data[i+self.img_num-1]['img']) + 0.1:
                for j in range(self.img_num):
                    del_list.append(i+j)
        
        data = [no_none_data[i] for i in range(len(no_none_data)) if i not in del_list]

        return data