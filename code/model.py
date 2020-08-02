import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LeaveOneOutModel:
    # create object storing path of data
    def __init__(self, X_train, X_test, Y_train, Y_test, model_name):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        self.X_train = self.scaler.transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.model_name = model_name

        self.bst_thresh, self._y_prob, self.fpr, self.tpr, self.thrshd_roc = self.leave_one_out_cv_v1(self.X_train, self.Y_train, self.model_name)

        self.clf = self.get_model(model_name, self.X_train, self.Y_train)        
    
    def leave_one_out_cv_v0(self, X, y, model_name):
        # choose threshold
        threshold_list = np.arange(0.01, 1, 0.01)
        score = np.zeros(threshold_list.shape)
        test_num = len(X)
        TP = np.zeros(threshold_list.shape)
        FN = np.zeros(threshold_list.shape)
        FP = np.zeros(threshold_list.shape)
        TN = np.zeros(threshold_list.shape)

        for i in range(len(threshold_list)):
            loo = LeaveOneOut()
            # leave one out loop
            for _train_index, _test_index in loo.split(X):
                _X_train, _X_test = X[_train_index], X[_test_index]
                _y_train, _y_test = y[_train_index], y[_test_index]
                clf = self.get_model(model_name, _X_train, _y_train)
                pred_proba_df = clf.predict_proba(_X_test)[:,1]
                if _y_test == 0:
                    if pred_proba_df <= threshold_list[i]:
                        score[i] += 1 / test_num
                        TN[i] += 1
                    else:
                        FN[i] += 1
                elif _y_test == 1:
                    if pred_proba_df > threshold_list[i]:
                        score[i] += 1 / test_num
                        TP[i] += 1
                    else:
                        FP[i] += 1
        # compute ROC
        # ######################
        # have error when denominator == 0
        TPR = TP / (TP + FN)
        FPR = TN / (TN + FP)
        # get the threshold of best score
        threshold = threshold_list[np.argmax(score)]

        return threshold, TPR, FPR

    def leave_one_out_cv_v1(self, X, y, model_name):

        # choose threshold
        threshold_list = np.arange(0.01, 1, 0.01)
        score = np.zeros(threshold_list.shape)
        test_num = len(X)
        _y_prob = np.zeros(len(X))

        loo = LeaveOneOut()
        # leave one out loop
        for _train_index, _test_index in loo.split(X):
            _X_train, _X_test = X[_train_index], X[_test_index]
            _y_train, _y_test = y[_train_index], y[_test_index]
            clf = self.get_model(model_name, _X_train, _y_train)
            pred_proba_df = clf.predict_proba(_X_test)[:,1]
            _y_prob[_test_index] = pred_proba_df
            for i in range(len(threshold_list)): 
                if _y_test == 0 and pred_proba_df <= threshold_list[i]:
                    score[i] += 1 / test_num
                elif _y_test == 1 and pred_proba_df > threshold_list[i]:
                    score[i] += 1 / test_num

        # get the threshold of best score
        threshold = threshold_list[np.argmax(score)]
        fpr, tpr, thrshd_roc = metrics.roc_curve(y, _y_prob, pos_label=1)
        # fpr, tpr, thrshd_roc = None, None, None

        return threshold, _y_prob, fpr, tpr, thrshd_roc                

    # bulid model
    def get_model(self, model_name, X_train, Y_train):
    
        # logistic regression
        if model_name == 'LR':
            clf = LogisticRegression(solver='lbfgs')
        # random forest
        elif model_name == 'RF':
            clf = RandomForestClassifier(max_depth=2, random_state=0)
        # C-Support Vector Classification
        elif model_name == 'SCV':
            clf = SVC(probability=True)
        # Multi-layer Perceptron
        elif model_name == 'MLP':
            clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(1000, 500), random_state=0)
        
        clf.fit(X_train, Y_train)

        return clf
    
    # see the metrics of model
    def get_metrics(self, thresh=None):

        if thresh == None:
            p = self.bst_thresh
        else:
            p = thresh

        pred_proba_df = pd.DataFrame(self.clf.predict_proba(self.X_test)[:,1])
        Y_pred = pred_proba_df.applymap(lambda x: 1 if x>p else 0).to_numpy().reshape((pred_proba_df.shape[0]))
        print("%s:\n%s\n" % (self.model_name,
            metrics.classification_report(self.Y_test, Y_pred)))
        return 0

    # get the indices of important features
    def get_important_feature(self):
        # logistic regression
        if self.model_name == 'LR':
            importance = self.clf.coef_[0]
        # random forest
        elif self.model_name == 'RF':
            importance = self.clf.feature_importances_

        return importance

    # false-positive rate
    def test_false_positive(self):
        # choose threshold
        pred_proba_df = pd.DataFrame(self.clf.predict_proba(self.X_test)[:,1])
        threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,  .95,.99]
        for i in threshold_list:
            print ('\n******** For i = {} ******'.format(i))
            Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0).to_numpy().reshape( (pred_proba_df.shape[0]))
            dataset = {'y_Actual':    self.Y_test,
                    'y_Predicted': Y_test_pred
                    }
            df = pd.DataFrame(dataset, columns=['y_Actual','y_Predicted'])
            confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames= ['Actual'], colnames=['Predicted'])
            plt.show()
            sn.heatmap(confusion_matrix, annot=True)

    # get the index of false-positive image
    def false_positive_index(self, clf, X_test, Y_test, threshold):
        pred_proba_df = pd.DataFrame(clf.predict_proba(X_test)[:,1])
        Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>threshold else 0).to_numpy().reshape( (pred_proba_df.shape[0]))
        false_positives = np.logical_and(Y_test != Y_test_pred, Y_test_pred == 1)
        return np.arange(len(Y_test))[false_positives]

    # get the index of false-negtive image
    def false_negtive_index(self, clf, X_test, Y_test, threshold):
        pred_proba_df = pd.DataFrame(clf.predict_proba(X_test)[:,1])
        Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>threshold else 0).to_numpy().reshape( (pred_proba_df.shape[0]))
        false_negtives = np.logical_and(Y_test != Y_test_pred, Y_test_pred == 0)
        return np.arange(len(Y_test))[false_negtives]