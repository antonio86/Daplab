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

def compute_fold(sc,data_tr,label_tr,data_ts, label_ts):
    i_models = sc.parallelize(range(1,16))
    # g = train_gmm(data_tr[label_tr == 1],1)
    models = i_models.map(lambda mod: train_gmm(data_tr[label_tr == mod],mod))
    models = models.collect()
    data_ts_par = sc.parallelize(data_ts)
    pred = data_ts_par.map(lambda ts: test_gmm(ts, models))
    return compute_acc_rate(pred.collect(),label_ts)

def train_gmm(data,mod):
    n_comp = 40
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
    sc = SparkContext("local", "Daplab II")
    Train_Test_data, Train_Test_label = read_data()
    kf = KFold(len(Train_Test_data),n_folds=2)
    samples = sc.parallelize(kf)
    g = mixture.GMM(n_components=2)
    Train_Test_data = np.array(Train_Test_data)
    Train_Test_label = np.array(Train_Test_label)
    sample = samples.collect()
    #gg = compute_fold(Train_Test_data[sample[0][0]], Train_Test_label[sample[0][0]],Train_Test_data[sample[0][1]], Train_Test_label[sample[0][1]])
    predictions = []
    for i in range(0,4):
        predictions.append(compute_fold(sc,Train_Test_data[sample[i][0]], Train_Test_label[sample[i][0]],Train_Test_data[sample[i][1]], Train_Test_label[sample[i][1]]))
    #predictions = samples.map(lambda fold: compute_fold(Train_Test_data[fold[0]], Train_Test_label[fold[0]],Train_Test_data[fold[1]], Train_Test_label[fold[1]]))
    print "****************************"
    print predictions
    print "****************************"
