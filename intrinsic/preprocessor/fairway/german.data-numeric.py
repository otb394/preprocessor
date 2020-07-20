import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import math
import numpy as np

def preprocess1(Y, X):
        index = []
        label = []
        # index_target = []
        for i in range(0, len(Y)):
                # y = Y[0][i]
                y = Y.iloc[i]
                if y == "close":
                        # y = "yes"
                        y = 1
                        # index_target.append(i)
                elif y == "open":
                        # y = "no"
                        y = 0
                elif y == "deleted":
                        index.append(i)  # index is a list of index for deleted samples
                label.append(y)

        for i in sorted(index, reverse=True):  # delete samples with deleted label
                del label[i]
                del X[i]

        # X_text = []                   # transfer samples of X from list to str pd.Series
        # for i in X:
        #           X_text.append(pd.Series(i).str.cat(sep=','))
        #           X = X_text  # list(type)
        return label, X

def get_files(path, extensions):
    if (len(extensions) == 0):
        return []
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if any(file.endswith('.' + ext) for ext in extensions):
            #if extension in file:
                files.append(os.path.join(r, file))
    return files

# The path should be a path to a csv
def data_prep(path):
        # Preprocessings:
        # 1. Remove extra white space
        # 2. Handle missing values. Either remove or fill
        # 3. Convert categorical into numeric as per one-hot or ordinal encoding
        # 4. Can try min-max scaling
        missing_values = ["?"]
        training_trim = pd.read_csv(path, na_values=missing_values, encoding='UTF-8', header=None)#encoding='ISO-8859-1')#
        print('NULL COLS')
        nul_col = training_trim.isna().any()
        nul_col.to_csv('temp_null_cols.csv')
        training_trim.dropna(inplace=True)

        training_x = training_trim.iloc[:, :-1]  # pandas.core.frame.DataFrame
        training_y = training_trim.iloc[:, -1]  # pandas.core.series.Series

        print(training_x.dtypes)
        orig_dim = len(training_x.columns)
        print('Original dim = ' + str(orig_dim))
        print('No. of instances = ' + str(training_x.shape[0]))
        cols = training_x.select_dtypes(include='object').columns
        print('cols = ' + str(cols))
        print('no. of cols = ' + str(len(cols)))
        if (len(cols) != 0):
            training_x = pd.concat([training_x, pd.get_dummies(training_x[cols], prefix=cols)], axis=1).drop(cols, axis=1)
        print(training_x.dtypes)

        # remove the samples in training and test set with label "deleted"
        training_y, training_x = preprocess1(training_y, training_x)


        # normalize the x for training and test sets
        min_max_scaler = preprocessing.MinMaxScaler()
        scaler = MinMaxScaler()

        #tot = training_x.shape[0]
        #sample_size = math.floor(tot * 1.0)
        #sample_size = math.floor(tot * 0.2)
        #training_x = training_x.head(n = sample_size)
        #training_x = training_x.iloc[16000:24000,:]
        print('Actual no. of rows = ' + str(training_x.shape[0]))
        num_inst = training_x.shape[0]
        columns = training_x.columns
        training_x = np.asarray(training_x)  # convert dataframe into numpy.array to normalize
        training_x = min_max_scaler.fit_transform(training_x)
        #print(training_x)
        return orig_dim, num_inst, pd.DataFrame(training_x, columns=columns)

name = 'german.data-numeric.csv'

path = 'data/fairway/' + name


orig_dim, num_inst, training_x = data_prep(path)

print(training_x)

training_x.to_csv('processed_data/fairway/processed_' + name, index=False)

