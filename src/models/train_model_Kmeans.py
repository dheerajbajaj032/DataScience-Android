import matplotlib
import os
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

columns = ["TESTId", "Native_Pss", "Native_Private_Dirty", "Native_Heap_Alloc", "Native_Heap_Free", "code_Pss",
           "code_Private_Dirty"]

pd.set_option('display.max_columns', 8)
script_path = os.path.dirname(__file__)
parent_file = os.path.join(script_path, os.pardir)

data_Android_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
data_Netflix_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')

df = pd.read_csv(data_Android_path, sep=',', index_col='TESTId', usecols=columns)

f1 = df['Native_Pss']
f2 = df['Native_Private_Dirty']
f3 = df['Native_Heap_Alloc']

X = np.matrix(zip(f1, f2, f3))
print X
kmeans = KMeans(n_clusters=2).fit(X)
T = kmeans.labels_
k = 3
centroid = {i+1 : kmeans.cluster_centers_
            for i in range(k)}

plt.scatter(f1, f2, f3, color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'}

for i in centroid.keys():
    plt.scatter(*centroid[i],
                color=colmap[i])
plt.show()