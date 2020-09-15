from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sn
from skimage import transform

class Visualize:
    
    
    # create object storing path of data
    def __init__(self):
        pass

    # plot roc curve
    def plot_roc_curve(self, fpr, tpr):
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    # plot confusion matrix
    def plot_confusion(self, clf, X_test, Y_test, threshold):
        pred_proba_df = pd.DataFrame(clf.predict_proba(X_test)[:,1])
        Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>threshold else 0).to_numpy().reshape((pred_proba_df.shape[0]))
        data = {'y_Actual': Y_test,
                'y_Predicted': Y_test_pred
                }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames= ['Actual'], colnames=['Predicted'])
        plt.show()
        figure = sn.heatmap(confusion_matrix, annot=True, fmt="d")
        return figure

    # plot n most important features 
    def plot_feature(self, importance, index, n):

        plt.figure()

        plt.bar(index[:n], importance[index][:n])
        plt.xlabel("the index of feature")
        plt.ylabel("importance")

        plt.show()
        

    def plot_image_by_feature_importance(self, importance, min_value, max_value, m=2, boxsize=3):
        image_type = ['past_dilate_img', 'future_dilate_img']
        img_size = (boxsize*2+1)**2
        for i in range(m):
            print('######################################################')
            print("plot feature importance on image", image_type[i])
            print('the importance of patch')
            feature_cut = importance[i*img_size:(i+1)*img_size].reshape(((boxsize*2+1), (boxsize*2+1)))
            ax = sn.heatmap(feature_cut, linewidth=0.5, vmin=min_value, vmax=max_value)
            plt.show()
            feature_cut_mean = importance[m*img_size+i]
            print('the importance of mean of patch is %f' % feature_cut_mean)
            feature_cut_std = importance[m*img_size+m+i]
            print('the importance of std of patch is %f' % feature_cut_std)
            print('the importance of edge detection of patch')
            feature_edge = importance[(m*img_size+2*m+i*img_size):(m*img_size+2*m+(i+1)*img_size)].reshape(((boxsize*2+1), (boxsize*2+1)))
            ax = sn.heatmap(feature_edge, linewidth=0.5, vmin=min_value, vmax=max_value)
            plt.show()
            feature_edge_mean = importance[m*img_size*2+2*m+i]
            print('the importance of mean of edge detection of patch is %f' % feature_edge_mean)
            feature_edge_std = importance[m*img_size*2+2*m+m+i]
            print('the importance of std of edge detection of patch is %f' % feature_edge_std)
            feature_ring_mean = importance[(m*img_size+2*m)*2+i]
            print('the importance of mean of ring patch is %f' % feature_ring_mean)
            feature_ring_std = importance[(m*img_size+2*m)*2+m+i]
            print('the importance of std of ring patch is %f' % feature_ring_std)
            elongation = importance[(m*img_size+2*m)*2+m+i+1]
            print('the importance of elongation is %f' % elongation)
            circularity = importance[(m*img_size+2*m)*2+m+i+2]
            print('the importance of circularity is %f' % circularity)
            compactness = importance[(m*img_size+2*m)*2+m+i+3]
            print('the importance of compactness is %f' % compactness)
