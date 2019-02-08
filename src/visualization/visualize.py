import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

columns = ["TESTId", "Native_Pss", "Native_Private_Dirty", "Native_Heap_Alloc", "Native_Heap_Free", "code_Pss",
           "code_Private_Dirty"]

pd.set_option('display.max_columns', 8)

script_path = os.path.dirname(__file__)
parent_file = os.path.join(script_path, os.pardir)

data_Android_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
data_Netflix_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')

df1 = pd.read_csv(data_Android_path, sep=',', index_col='TESTId', usecols=columns)
plt.figure(figsize=(20, 10))
plt.title("Airtel OTT")
df1.boxplot()

df2 = pd.read_csv(data_Netflix_path, sep=',', index_col='TESTId', usecols=columns)
plt.figure(figsize=(20, 10))
plt.title("Netflix")
df2.boxplot()

plt.show()
