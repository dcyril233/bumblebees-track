import numpy as np
import glob
import pickle


class Image:
    
    
    # create object storing path of data
    def __init__(self, path):
        self.path = path


    # import data
    def import_data(self, del_num=0):
        old_data = []
        del_pair =[]
        for fn in sorted(glob.glob(self.path)):
            dat = pickle.load(open(fn,'rb'))
            if dat[1] is not None:
                dat[1] = dat[1].astype(np.float)
            else:
                del_pair.append(dat[0]//3)
            old_data.append(dat)
        if del_num != 0:
            del old_data[:del_num]

        del_list = list(map(lambda x : 3*x, del_pair)) + list(map(lambda x : 3*x + 1, del_pair)) + list (map(lambda x : 3*x + 2, del_pair))
        data = [image for image in old_data if image[0] not in del_list]

        return data