__author__ = 'antonioridi'

#import os
#import sys
#import numpy as np
#os.environ['SPARK_HOME'] = "/Users/antonioridi/Documents/spark-1.5.2"
#sys.path.append("/Users/antonioridi/Documents/spark-1.5.2/python")

#try:
#    from pyspark import SparkContext
#    from pyspark import SparkConf
#    from sklearn.cross_validation import train_test_split, KFold
#    from sklearn import mixture
#    from sklearn.datasets import make_classification
#    from sklearn.metrics import accuracy_score
#    from sklearn.tree import DecisionTreeClassifier
#except ImportError as e:
#    print ("Error importing Spark Modules", e)
#    sys.exit(1)

import numpy as np
from pyspark import SparkContext
from sklearn.cross_validation import KFold
from sklearn import mixture
import time

def read_data():
    Train_Test_data = []
    Train_Test_label = []
    file_data = open("Train_Test_data.csv","rb")
    file_label = open("Train_Test_label.csv","rb")
    lines = file_data.readlines()
    lines_label = file_label.readlines()
    for i in range(0,len(lines_label)):
        c = []
        for j in range(0,6):
            c_str = lines[i*6+j].split(',')
            c_float = [float(x) for x in c_str]
            c.append(c_float)
        labels = lines_label[i].split(',')
        Train_Test_data.append(c)
        Train_Test_label.append(int(labels[1]))
    return Train_Test_data, Train_Test_label

def train_gmm(data,mod):
    n_comp = 250
    g = mixture.GMM(n_components=n_comp)
    data_tr = []
    for j in range(0,6):
        data_tr.append(np.hstack(np.dstack(data)[0][j]))
    label_tr = np.repeat(mod,len(data_tr[0]))
    g.fit(np.transpose(data_tr),label_tr)
    return g

def test_gmm(ts, models):
    class_score = []
    for j in range(0,15):
        class_score.append(np.sum(models[j].score(np.transpose(np.vstack(ts)))))
    pos_win = class_score.index(max(class_score))
    return pos_win

def compute_acc_rate(predicted,ground_truth):
    Conf_mat = np.zeros([15,15])
    for i in range(0,len(ground_truth)):
        Conf_mat[predicted[i]][ground_truth[i]-1] += 1
    return np.trace(Conf_mat)/np.sum(np.sum(Conf_mat))

if __name__ == '__main__':
    start_time = time.time()
    sc = SparkContext("local", "Daplab III")
    nFolds = 4
    Train_Test_data, Train_Test_label = read_data()
    kf = KFold(len(Train_Test_data),n_folds=nFolds)
    samples = sc.parallelize(kf)
    g = mixture.GMM(n_components=2)
    Train_Test_data = np.array(Train_Test_data)
    Train_Test_label = np.array(Train_Test_label)
    sample = samples.collect()

    parallel_fold = []
    for i in range(0,nFolds):
        tmp_idx_tr = sample[i][0]
        tmp_idx_ts = sample[i][1]
        for j in range(1, 16):
            tmp_par = []
            tmp_par.append(tmp_idx_tr[Train_Test_label[tmp_idx_tr] == j])
            tmp_par.append(tmp_idx_ts[Train_Test_label[tmp_idx_ts] == j])
            parallel_fold.append(tmp_par)

    folds = sc.parallelize(parallel_fold)
    models = folds.map(lambda f: train_gmm(Train_Test_data[f[0]], Train_Test_label[f[0]][0]))
    model = models.collect()
    AccRate = []
    for i in range(0,nFolds):
        data_ts_par = sc.parallelize(Train_Test_data[sample[i][1]])
        sub_models = model[i*15:i*15+15]
        pred = data_ts_par.map(lambda ts: test_gmm(ts, sub_models))
        AccRate.append(compute_acc_rate(pred.collect(),Train_Test_label[sample[i][1]]))

    print "****************************"
    print(AccRate)
    print "****************************"
    print("--- %s seconds ---" % (time.time() - start_time))