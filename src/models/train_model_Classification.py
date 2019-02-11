import matplotlib
import os
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


columns = ["TESTId", "Native_Pss", "Native_Private_Dirty", "Native_Heap_Alloc", "Native_Heap_Free", "code_Pss",
           "code_Private_Dirty"]
d = {"SampleTest_1": 1, "Airtel_PinScreen": 2, "Airtel_HomeScreen": 3, "Airtel_DetailScreen": 4,
     "Airtel_PlayerScreen": 5,
     "Airtel_Scrolling": 6, "Airtel_SearchScreen": 7}
e = {"Netflix_PinScreen": 2, "Netflix_HomeScreen": 3, "Netflix_DetailScreen": 4,
     "Netflix_PlayerScreen": 5,
     "Netflix_Scrolling": 6, "Netflix_SearchScreen": 7}
pd.set_option('display.max_columns', 8)
script_path = os.path.dirname(__file__)
parent_file = os.path.join(script_path, os.pardir)
data_Android_path = os.path.join(parent_file, 'data', 'ANDROID.csv')
data_Netflix_path = os.path.join(parent_file, 'data', 'NETFLIX.csv')

# -------------------------------Creating Android train X and Y ------------------------------

df = pd.read_csv(data_Android_path, sep=',', usecols=columns)
index_column_unique = df['TESTId'].unique()
df['TESTId'] = df['TESTId'].map(d, na_action='ignore')
# Plot the count of each unique label
sns.countplot(df['TESTId'], label='Count')
# Boxplot the Input feature numeric variable
df.drop('TESTId', axis=1).plot(kind='box', subplots=True, layout=(3, 2), sharex=True,
                               sharey=True, figsize=(9, 9))
plt.title = 'Box Plot for each input variable'
# Histogram of Input features numeric variable
df.drop('TESTId', axis=1).hist(bins=20, figsize=(9, 9))
plt.title = 'Histogram for each input variable'
# Scatter Matrix for each input variable
X_train_sub = df[columns[1:]]
y_train = df['TESTId'].values
cmap = plt.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train_sub, c=y_train, marker='o', s=40,
                            hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
plt.suptitle('Scatter-matrix for each input variable')
# plt.savefig('Android_Box')

# -------------------------------Creating Test X and Y  --------------------------------------------------------
df_test = pd.read_csv(data_Netflix_path, sep=',', usecols=columns)
df_test['TESTId'] = df_test['TESTId'].map(e, na_action='ignore')
X_test = df_test[columns[1:]]
y_test = df_test['TESTId'].values

# -------------------------------Creating Training and Test Sets and Apply Scaling ------------------------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_sub)
X_test = scaler.transform(X_test)

# -------------------------------Model---------------------------------------------------------------------------
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print "Accuracy of k-nn classifier on training set: {:.2f}".format(knn.score(X_train, y_train))
print "Accuracy of k-nn classifier on test set: {:.2f}".format(knn.score(X_test, y_test))

# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# print "Accuracy of logistic reg classifier on training set: {:.2f}".format(logreg.score(X_train, y_train))
# print "Accuracy of logistic reg on test set: {:.2f}".format(logreg.score(X_test, y_test))
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

#plt.show()

