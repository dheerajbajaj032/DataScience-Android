import warnings

warnings.simplefilter(action='ignore', category=Warning)
import matplotlib
import os

matplotlib.use('TkAgg')
import pandas as pd
from resource import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Configurations
pd.set_option('display.max_columns', 8)


class StdMeanClass:

    def __init__(self):
        self.scaler = preprocessing.MinMaxScaler()

    def create_boxplot(self, path, rows):
        df_sub = pd.read_csv(path, sep=',', usecols=columns)
        df_sub['TESTId'] = df_sub['TESTId'].map(rows, na_action='ignore')
        #print df_sub
        df_sub = self.scaler.fit_transform(df_sub)
        #print df_sub
        df_sub_2d = np.array(df_sub)
        x = pd.DataFrame(df_sub_2d, columns=columns)
        x = x.drop(['TESTId'], axis=1)
        #print x
        return x, x.describe()

    def box_plot(self, train, test):
        plt.subplot(2, 1, 1)
        train.boxplot(patch_artist=True)
        plt.title('Training')

        plt.subplot(2, 1, 2)
        test.boxplot(patch_artist=True)
        plt.title('Test')

    def std_mean_plot(self, name, index, df_train, df_test):
        sdict = {}
        for i in range(0, 6):
            sdict[columns[i + 1]] = [df_train.iloc[index, i], df_test.iloc[index, i]]
        df = pd.DataFrame.from_dict(sdict)
        df_transpose = df.T
        df_transpose = df_transpose.rename(index=str, columns={0: "train", 1: "test"})
        if name == 'Mean':
            df_transpose['diff'] = df_transpose['test'] - df_transpose['train']
            print name + " diff b/w train and test (This should be as less as possible): " + str(
                df_transpose['diff'].mean())
        else:
            df_transpose['diff'] = df_transpose['train'] - df_transpose['test']
            print name + " diff b/w train and test (The bigger the deviation better the performance): " + str(
                df_transpose['diff'].mean())

        df_transpose.plot.bar()
        plt.title(name + ' deviation b/w test and train')


# Load Data to create Dataframe

script_path = os.path.dirname(__file__)
parent_file = os.path.join(script_path, os.pardir)
test_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
train_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')

# Initialize class
obj = StdMeanClass()

# Create boxplot
train_x, desc_train_x = obj.create_boxplot(train_path, train_rows)
test_x, desc_test_x = obj.create_boxplot(test_path, test_rows)
obj.box_plot(train_x, test_x)

# Create Std plot
obj.std_mean_plot(name='Std', index=2, df_test=desc_test_x, df_train=desc_train_x)

# Create Mean plot
obj.std_mean_plot(name='Mean', index=1, df_test=desc_test_x, df_train=desc_train_x)

plt.show()
