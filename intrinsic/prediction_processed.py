import numpy as np
import glob, os, sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import time
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pdb
from estimate_intrinsic_dim import Solver as IntrinDimSolver
from utils import ConfigBase
from sklearn.preprocessing import OneHotEncoder
import math
import ntpath

""" NOTES:
            - requires Python 3.0 or greater
"""

class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()
        self.est_config = "="*6 + " Intrinsic Dim Estimation of real dataset " + "="*6
        self.log_dir = "exp/realcase/log"
        self.log_filename = "est.log"
        #self.logrs = "-4:0:20"  # from -4 to 0 with 20 steps
        #self.logrs = "-4:0:10"  # from -4 to 0 with 20 steps
        self.logrs = "-10:10:100"  # from -10 to 10 with 100 steps
        #self.logrs = "-10:10:200"  # from -10 to 10 with 100 steps
        

class DatasetWrapper():
    def __init__(self, dataset, config):
        self.samples = dataset

    def __len__(self):
        return self.samples.shape[0]

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

def main(config):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        projects = ['seacraft']
        for project in projects:
                start_time = time.time()
                # path = r'data/total_features/' + project
                #path = r'data/' + project
                path = r'processed_data/' + project
                print("-----------------" + project + "----------------------")
                AUC = []
                cost = []
                Acc = []
                #path = path + '/subject_systems'
                files = get_files(path, ['csv'])
                #files = [path + '/processed_compas-scores-two-years.csv']
                #files = [path + '/default.csv']
                #exception_list = ['processed_data/fairway/processed_compas-scores-two-years.csv']
                #exception_set = set(exception_list)
                #files = [x for x in files if x not in exception_set]
                print('Files:')
                print(files)

                file_names = []
                for f in files:
                    basename = ntpath.basename(f)
                    fileName, fileExt = os.path.splitext(basename)
                    file_names.append(fileName)

                ans = []
                for i in range(len(files)):
                    try:
                        print('i = ' + str(i))
                        fil = files[i]
                        name = file_names[i]
                        print('fil = ' + str(fil))
                        print('name = ' + str(name))

                        training_data = pd.read_csv(fil)
                        num_inst = training_data.shape[0]
                        orig_dim = training_data.shape[1]

                        # Sampling
                        #sample_size = min(2000, num_inst)
                        #training_data = training_data.head(n = sample_size)
                        print('Original no. of instances = ' + str(num_inst))
                        #num_inst = sample_size

                        #print("Training data=")
                        #print(training_data)
                                                                
                        # intrinDimEstimation
                        training_data = np.asarray(training_data)
                        dataset = DatasetWrapper(training_data, config)
                        print(">> creating intrinsic dimension solver")
                        solver = IntrinDimSolver(dataset, config)
                        print(">> solving...")
                        #solver.show_curve(config.logrs)
                        val = solver.show_curve(config.logrs)
                        val2 = solver.show_curve(config.logrs, version = 2)
                        print(">> task finished")
                        if (name.startswith('processed_')):
                            name = name[10:]
                        ret = [name, orig_dim, val, val2, num_inst]
                        ans.append(ret)
                    except:
                        print(str(file_names[i]) + ' threw error')


                ans_df = pd.DataFrame(ans, columns=['Dataset', 'original dim', 'intrinsic dim_L1', 'intrinsic dim_L2', 'No. of instances'])
                ans_df.to_csv('calculator_output/' + project + '_output.csv', index=False)

if __name__ == "__main__":
    from utils import Logger
    config = Config()
    config.parse_args()
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    sys.stdout = Logger('{0}/{1}'.format(config.log_dir, config.log_filename))
    config.print_args()
    main(config)
