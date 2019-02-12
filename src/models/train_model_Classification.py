import matplotlib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from src.models.resource import *


# -------------------------------Creating Android train X and Y ------------------------------

class PerformanceDataClass:

    def __init__(self):
        pd.set_option('display.max_columns', 8)
        script_path = os.path.dirname(__file__)
        parent_file = os.path.join(script_path, os.pardir)
        self.data_test_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
        self.data_train_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')

    def getTrainValues(self):
        print self.data_train_path
        df_train = pd.read_csv(self.data_train_path, sep=',', usecols=columns)
        # index_column_unique = df['TESTId'].unique()
        df_train['TESTId'] = df_train['TESTId'].map(train_rows, na_action='ignore')
        x_train_sub = df_train[columns[1:]]
        y_train = df_train['TESTId'].values

        return x_train_sub, y_train

    def getTestValues(self):
        df_test = pd.read_csv(self.data_test_path, sep=',', usecols=columns)
        # index_column_unique = df['TESTId'].unique()
        df_test['TESTId'] = df_test['TESTId'].map(test_rows, na_action='ignore')
        x_test = df_test[columns[1:]]
        y_test = df_test['TESTId'].values
        return x_test, y_test


class PerformanceModel(PerformanceDataClass):

    def __init__(self):
        PerformanceDataClass.__init__(self)
        self.obj = PerformanceDataClass()

    def scaling_ModelPrediction(self):
        scalar = MinMaxScaler()
        x_train, y_train = self.obj.getTrainValues()
        x_test, y_test = self.obj.getTestValues()
        x_train_final = scalar.fit_transform(x_train)
        x_test_final = scalar.transform(x_test)
        decision_clf = DecisionTreeClassifier()
        decision_clf.fit(x_train_final, y_train)
        ypred_train = decision_clf.predict(x_train_final)
        ypred_test = decision_clf.predict(x_test_final)

        print "Accuracy of Decision Tree classifier on training set: {:.2f}".format(
            accuracy_score(y_train, ypred_train))
        print "Accuracy of Decision Tree classifier on test set: {:.2f}".format(accuracy_score(y_test, ypred_test))

        print confusion_matrix(y_train, ypred_train)
        print confusion_matrix(y_test, ypred_test)


print PerformanceModel().scaling_ModelPrediction()

# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# print "Accuracy of k-nn classifier on training set: {:.2f}".format(knn.score(X_train, y_train))
# print "Accuracy of k-nn classifier on test set: {:.2f}".format(knn.score(X_test, y_test))
#
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# print "Accuracy of Decision Tree classifier on training set: {:.2f}".format(clf.score(X_train, y_train))
# print "Accuracy of Decision Tree on test set: {:.2f}".format(clf.score(X_test, y_test))
#
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# print "Accuracy of LDA classifier on training set: {:.2f}".format(gnb.score(X_train, y_train))
# print "Accuracy of LDA on test set: {:.2f}".format(gnb.score(X_test, y_test))
#
# svm = SVC()
# svm.fit(X_train, y_train)
# print "Accuracy of svm classifier on training set: {:.2f}".format(svm.score(X_train, y_train))
# print "Accuracy of svm on test set: {:.2f}".format(svm.score(X_test, y_test))

# plt.show()
