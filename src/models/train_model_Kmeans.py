import warnings

warnings.simplefilter(action='ignore', category=Warning)
import matplotlib
import os
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')
import pandas as pd
from resource import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 8)
# Load Data to create Dataframe
script_path = os.path.dirname(__file__)
parent_file = os.path.join(script_path, os.pardir)

data_test_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
data_train_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')

df_train = pd.read_csv(data_train_path, sep=',', usecols=columns)
df_train['TESTId'] = df_train['TESTId'].map(train_rows, na_action='ignore')
np_array_2D = np.array(df_train)
df_2D = pd.DataFrame(np_array_2D, columns=columns)

# # K-Means Model
'''
k_rng = range(1, 10)
sse = []
for k in k_rng:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_train[['Native_Pss', 'Native_Private_Dirty',
                         'Native_Heap_Alloc', 'Native_Heap_Free',
                         'code_Pss', 'code_Private_Dirty']])
    sse.append(kmeans.inertia_)

print sse
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()
'''

# Scaling
scaler = MinMaxScaler()
df_2D[['Native_Pss', 'Native_Private_Dirty', 'Native_Heap_Alloc', 'Native_Heap_Free',
       'code_Pss', 'code_Private_Dirty']] = scaler.fit_transform(df_2D[['Native_Pss',
                                                                        'Native_Private_Dirty', 'Native_Heap_Alloc',
                                                                        'Native_Heap_Free', 'code_Pss',
                                                                        'code_Private_Dirty']])
km = KMeans(n_clusters=6)
y_predicted = km.fit_predict(df_2D[['Native_Private_Dirty',
                                    'Native_Heap_Alloc']])
# y_predicted_3D = km.fit_predict(df_2D[['Native_Pss', 'Native_Private_Dirty',
#                                       'Native_Heap_Alloc']])
# Cluster Centroids
centroids = km.cluster_centers_
# print y_predicted
df_2D['cluster'] = y_predicted

df1 = df_2D[df_2D.cluster == 0]
df2 = df_2D[df_2D.cluster == 1]
df3 = df_2D[df_2D.cluster == 2]
df4 = df_2D[df_2D.cluster == 3]

# plt.scatter(df1['Native_Private_Dirty'], df1['Native_Heap_Alloc'], color='green')
# plt.scatter(df2['Native_Private_Dirty'], df2['Native_Heap_Alloc'], color='green')
# plt.scatter(df3['Native_Private_Dirty'], df3['Native_Heap_Alloc'], color='green')
# plt.scatter(df4['Native_Private_Dirty'], df4['Native_Heap_Alloc'], color='green')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='green', marker='*', label='Centroid Train')
plt.xlabel('Native_Private_Dirty')
plt.ylabel('Native_Heap_Alloc')
plt.legend()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(df1['Native_Private_Dirty'], df1['Native_Heap_Alloc'], df1['Native_Pss'], color='green', marker='o')
# ax.scatter(df2['Native_Private_Dirty'], df2['Native_Heap_Alloc'], df2['Native_Pss'], color='red', marker='o')
# ax.scatter(df3['Native_Private_Dirty'], df3['Native_Heap_Alloc'], df3['Native_Pss'], color='black', marker='o')
# ax.scatter(df4['Native_Private_Dirty'], df4['Native_Heap_Alloc'], df4['Native_Pss'], color='blue', marker='o')
# ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 1], color='purple', marker='*',
#            label='Centroid')
# ax.set_xlabel('Native_Private_Dirty')
# ax.set_ylabel('Native_Heap_Alloc')
# ax.set_zlabel('Native_Pss')

# Test set ------------------------------

df_test = pd.read_csv(data_test_path, sep=',', usecols=columns)
df_test['TESTId'] = df_test['TESTId'].map(train_rows, na_action='ignore')
np_array_2D_test = np.array(df_test)
df_2D_test = pd.DataFrame(np_array_2D_test, columns=columns)

# Scaling
scaler = MinMaxScaler()
df_2D_test[['Native_Pss', 'Native_Private_Dirty', 'Native_Heap_Alloc', 'Native_Heap_Free',
            'code_Pss', 'code_Private_Dirty']] = scaler.fit_transform(df_2D_test[['Native_Pss',
                                                                                  'Native_Private_Dirty',
                                                                                  'Native_Heap_Alloc',
                                                                                  'Native_Heap_Free', 'code_Pss',
                                                                                  'code_Private_Dirty']])

km_3d = KMeans(n_clusters=6)
y_predicted = km.fit_predict(df_2D_test[['Native_Private_Dirty',
                                         'Native_Heap_Alloc']])

# y_predicted_3D = km_3d.fit_predict(df_2D_test[['Native_Pss', 'Native_Private_Dirty',
#                                            'Native_Heap_Alloc']])

centroids_test = km.cluster_centers_
# print y_predicted
df_2D_test['cluster'] = y_predicted

df1_test = df_2D_test[df_2D_test.cluster == 0]
df2_test = df_2D_test[df_2D_test.cluster == 1]
df3_test = df_2D_test[df_2D_test.cluster == 2]
df4_test = df_2D_test[df_2D_test.cluster == 3]

# plt.scatter(df1_test['Native_Private_Dirty'], df1_test['Native_Heap_Alloc'], color='red')
# plt.scatter(df2_test['Native_Private_Dirty'], df2_test['Native_Heap_Alloc'], color='red')
# plt.scatter(df3_test['Native_Private_Dirty'], df3_test['Native_Heap_Alloc'], color='red')
# plt.scatter(df4_test['Native_Private_Dirty'], df4_test['Native_Heap_Alloc'], color='red')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='red', marker='*', label='Centroid Test')
plt.legend()
# Euclidean distance
diff = centroids-centroids_test
dist = np.linalg.norm(centroids - centroids_test)

print dist
print diff

plt.show()
