__author__ = 'antonioridi'


import numpy as np
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
    n_comp = 256
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
    nFolds = 4
    Train_Test_data, Train_Test_label = read_data()
    g = mixture.GMM(n_components=2)
    Train_Test_data = np.array(Train_Test_data)
    Train_Test_label = np.array(Train_Test_label)
    kf = KFold(len(Train_Test_data),n_folds=nFolds)
    AccRate = []
    for train_index, test_index in kf:
        Model = []
        for i in range(1,16):
            fold_model_idx = train_index[Train_Test_label[train_index] == i]
            Model.append(train_gmm(Train_Test_data[fold_model_idx], Train_Test_label[fold_model_idx][0]))
        pred = []
        for ts in Train_Test_data[test_index]:
            pred.append(test_gmm(ts,Model))
        AccRate.append(compute_acc_rate(pred,Train_Test_label[test_index]))

    print "****************************"
    print(AccRate)
    print "****************************"
    print("--- %s seconds ---" % (time.time() - start_time))