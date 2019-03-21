import matplotlib
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=Warning)
import pandas as pd
import seaborn as sns
from src.models.resource import *

columns = ["TESTId", "Native_Pss", "Native_Private_Dirty", "Native_Heap_Alloc", "Native_Heap_Free", "code_Pss",
           "code_Private_Dirty"]

pd.set_option('display.max_columns', 8)

script_path = os.path.dirname(__file__)
parent_file = os.path.join(script_path, os.pardir)

data_Android_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
data_Netflix_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')

df1 = pd.read_csv(data_Android_path, sep=',', index_col='TESTId', usecols=columns)
#print df1

plt.figure(figsize=(20, 10))
plt.title("Airtel OTT")
#df1.boxplot()

df2 = pd.read_csv(data_Netflix_path, sep=',', index_col='TESTId', usecols=columns)
plt.figure(figsize=(20, 10))
plt.title("Netflix")
#df2.boxplot()

# Plot the count of each unique label
df_train = pd.read_csv(data_Netflix_path, sep=',', usecols=columns)
print df_train
sns.countplot(df_train['TESTId'], label='Count')
plt.show()

# # Boxplot the Input feature numeric variable
# df.drop('TESTId', axis=1).plot(kind='box', subplots=True, layout=(3, 2), sharex=True,
#                                sharey=True, figsize=(9, 9))
# plt.title = 'Box Plot for each input variable'
# # Histogram of Input features numeric variable
# df.drop('TESTId', axis=1).hist(bins=20, figsize=(9, 9))
# plt.title = 'Histogram for each input variable'
# Scatter Matrix for each input variable

# cmap = plt.get_cmap('gnuplot')
# scatter = pd.scatter_matrix(X_train_sub, c=y_train, marker='o', s=40,
#                             hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
# plt.suptitle('Scatter-matrix for each input variable')
# # plt.savefig('Android_Box')


# data_test_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
# data_train_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')
#
# df_train = pd.read_csv(data_train_path, sep=',', usecols=columns)
# df_train['TESTId'] = df_train['TESTId'].map(train_rows, na_action='ignore')
# # df_train.drop('TESTId', axis=1).plot(kind='box', subplots=True, layout=(3, 2), sharex=True,
# #                                      sharey=True, figsize=(9, 9))
# # Create X and y
# X = df_train.drop(['TESTId'], 1).astype(int)
# y = np.array(df_train['TESTId'])
# cmap = plt.get_cmap('gnuplot')
# # scatter = pd.scatter_matrix(X, c=y, marker='o', s=40,
# #                             hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
# # plt.show()
