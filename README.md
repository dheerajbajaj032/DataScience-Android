AndroidPerformance
==============================

Predicting Android build Performance using Machine Learning

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template

Android Application performance metrics include ( Memory, CPU, Network, Systrace Analysis and many more .. ) Out of all memory is most important metric and can be used to create initial data

Get data from Phone using Memory dumpsys ( Adb shell dumpsys meminfo --Package name)
Save data in database (Sqlite3)
Read data from db, convert into pandas dataframe
Get data from Android :

Command : adb shell dumpsys meminfo package_name|pid
The output lists all of your app's current allocations, measured in kilobytes

Save data in database (Sqlite3):
Every android app contains multiple activities , For Android app performance analysis we should check memory consumption for each activity. Generally apps contain following screens :
1. Pin Screen 
2. Home Screen 
3. Detail Screen
4. Search Screen
and many more ...

Now in terms of Machine Learning we can use each screen as an unique class and Memory details ( Native_Pss , Native_Private_Dirty , Native_Heap_Alloc, Native_Heap_Free, Code_Pss and Code_Private_Dirty ) as Features

Next Step is to access db file and create a dataframe. How to do that ?
We will access db using Pandas ( read_sql_query built_in func)
def fetchrows(self):
    df = pd.read_sql_query("SELECT * FROM ANDROID", self.conn)
    df.set_index('TESTId')
    df.to_csv("ANDROID.csv")
    return df

Here "SELECT * FROM ANDROID" is a basic sql query
ANDROID is Table name
self.conn is an object pointing out to the database connection instance:
def __init__(self, dbname):
    self.conn = sqlite3.connect(dbname)
    print "Database connected successfully"

We can also simply convert dataframe to csv file using df.to_csv(Filename)
df_train = pd.read_csv(self.data_train_path, sep=',', usecols=columns)
Here we will have to specify all columns we want to use. If we don't specify, df will include index as a column

TESTId column contains string values. We can convert each instance to an integer
df_train['TESTId'] = df_train['TESTId'].map(train_rows, na_action='ignore') where train_rows
is a python dictionary of keys TESTId's and respective integer mapping
train_rows = {"PinScreen": 2, "HomeScreen": 3, "DetailScreen": 4,"PlayerScreen": 5,"Scrolling": 6, "SearchScreen": 7}

Dataframe output looks like :
Finally splitting dataframe into Xtrain & Ytrain :
X_train = df_train[columns[1:]]
y_train = df_train['TESTId'].values

    
    </a>. #cookiecutterdatascience</small>
</p>
