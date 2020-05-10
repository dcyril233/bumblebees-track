from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sn

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

        return 0


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
        sn.heatmap(confusion_matrix, annot=True, fmt="d")
        return 0


    # plot n most important features 
    def plot_feature(self, importance, index, n):

        plt.figure()

        plt.bar(index[:n], importance[index][:n])
        plt.xlabel("the index of feature")
        plt.ylabel("importance")

        plt.show()
        
        return 0