"""
Name: PYF_Train
Description: Trains lightweight prediction models for targets POE and WOP from the Preprocessed dataset
"""
import sys
import os
import os.path as path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
import joblib

from PYF_Model import LWModel
################################################
# Setup constants and paths
################################################
USR      = sys.argv[1]
LND      = sys.argv[2]
MTR   = sys.argv[3]
QNT = sys.argv[4]
VT_SPLIT = float(sys.argv[5])
KFOLD    = int(sys.argv[6])

QUANTILES = ['50', '75', '90']
TARGET_VARS = {
    'POE': ['POE'],
    'WOP': ['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95']
}

LABELS = ['LOW', 'MID', 'HIGH']
WOP_BINS = [-1, 250, 1500, 2000]
POE_BINS = [-1, 0.333, 0.66, 2]

MODEL_FILE_FMT = '{}_{}.model'
FILE_NAME_FMT = 'HLT_{0}_{1}_qnt.csv'
BASE_DIR = LND
CLN_DATA_DIR_PATH = path.join(BASE_DIR, 'Clean')
MODEL_DIR_PATH = path.join(BASE_DIR, 'Model')
GRAPHS_DIR_PATH = path.join(BASE_DIR, 'Graphics')

if (not path.isdir(GRAPHS_DIR_PATH)):
    os.mkdir(GRAPHS_DIR_PATH)
################################################
# Training and Evaluation Functions
################################################
def c_discrete_error(X, y, y_pred, class_labels):
  n_correct = sum(y_pred == y)
  return confusion_matrix(y_true=y, y_pred=y_pred, labels=class_labels, normalize='true')

def d_discrete_error(conf_mat, class_labels, task):
  disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_labels)
  plt.figure(figsize=(16,14))
  disp.plot(values_format = '.5g')
  plt.title("{} Conf Matrix".format(task))
  plt.savefig(path.join(GRAPHS_DIR_PATH, 'g_{}_{}.jpg'.format(USR, task)))

def correct_classes(classes):
    list_classes = list(classes)
    if 'MID' not in list_classes:
        return np.array(list_classes + ['MID'])
    else:
        return classes

def correct_conf_mat(conf_mat):
    if len(conf_mat) == 2:
        return np.array([
            [conf_mat[0,0], conf_mat[0,1], 0.0],
            [conf_mat[1,0], conf_mat[1,1], 0.0],
            [0.0          , 0.0          , 1.0]
        ], dtype=np.float64)
    else:
        return conf_mat

def folds_training(dataX, dataY, newClfFn, target_name, pretty_target_name):
    best_clf = -1

    train_err = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=np.float64)
    test_err  = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=np.float64)

    kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=42)
    for i, (train_ind, test_ind) in enumerate(kf.split(dataX, dataY)):
        print("Doing {} fold".format(i))
        X_train, X_test = dataX[train_ind], dataX[test_ind]
        y_train, y_test = dataY[train_ind], dataY[test_ind]

        clf = newClfFn()
        clf.fit(X_train, y_train)
        classes = correct_classes(clf.classes_)

        y_pred = clf.predict(X_train)
        print(classes)
        train_conf_mat = correct_conf_mat(c_discrete_error(X_train, y_train, y_pred, classes))
        print('Train Err')
        print(train_conf_mat)
        y_pred = clf.predict(X_test)
        test_conf_mat = correct_conf_mat(c_discrete_error(X_test, y_test, y_pred, classes))
        print('Test Err')
        print(test_conf_mat)

        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                train_err[i][j] += train_conf_mat[i][j]
                test_err[i][j] += test_conf_mat[i][j]

        if best_clf == -1 or True:
            best_clf = clf

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            train_err[i][j] /= KFOLD
            test_err[i][j]  /= KFOLD

    classes = correct_classes(clf.classes_)
    print('--------------------------')
    print(pretty_target_name)
    print('Classes: '+str(clf.classes_))
    print('- Train')
    # print('-- TP: ' ())
    print(train_err)
    print('- Test')
    print(test_err)
    print('--------------------------')
    d_discrete_error(train_err, classes, '{} - Train'.format(target_name))
    d_discrete_error(test_err, classes, '{} - Test'.format(target_name))

    return best_clf
################################################
# Train model
################################################
## Train WOPs
WOPS_CLFS = {}
POE_CLF = -1

MTR = 'WOP'
for qnt in QUANTILES:
    df = pd.read_csv(path.join(CLN_DATA_DIR_PATH, FILE_NAME_FMT.format(MTR, qnt)))
    for target in TARGET_VARS['WOP']:
        print('Model for WOP[{}] in qnt {}%'.format(target, qnt))
        discrete_target = target + '_D'
        df[discrete_target] = pd.cut(df[target],
                                    bins = WOP_BINS,
                                    labels=LABELS)
        dataX = df[['i_pop', 'i_ren' , 'i_res', 'i_mad', 'i_mat', 'i_nmosquitos']].copy().to_numpy(np.float64)
        dataY = df[discrete_target].copy().to_numpy()
        target_name = 'WOP_{}_{}'.format(target, qnt)
        clf = folds_training(dataX, dataY,
                        # lambda: KNeighborsClassifier(n_neighbors=3),
                        lambda: OneVsOneClassifier(LinearSVC(random_state=42)),
                        target_name, 'WOP[{}]_{}%'.format(target, qnt))

        WOPS_CLFS[target_name] = clf

## Train POE
MTR = 'POE'
target = 'POE'
qnt = 90
df = pd.read_csv(path.join(CLN_DATA_DIR_PATH, FILE_NAME_FMT.format(MTR, qnt)))
print('Model for POE in qnt {}%'.format(target, qnt))
discrete_target = target + '_D'
df[discrete_target] = pd.cut(df[target],
                            bins = POE_BINS,
                            labels=LABELS)
dataX = df[['i_pop', 'i_ren' , 'i_res', 'i_mad', 'i_mat', 'i_nmosquitos']].copy().to_numpy(np.float64)
dataY = df[discrete_target].copy().to_numpy()
POE_CLF = folds_training(dataX, dataY,
                lambda: DecisionTreeClassifier(random_state=42, max_depth=5),
                'POE_{}'.format(qnt), 'POE'.format(target, qnt)) 
################################################
# Ensemble Models into single model
################################################


model = LWModel(WOPS_CLFS, POE_CLF)
# print(model.predict([32, 18, 100, 25, 30])['poe'])
################################################
# Save built model
################################################
if (not path.isdir(MODEL_DIR_PATH)):
    os.mkdir(MODEL_DIR_PATH)


model_name = 'lw_pyf_model'.format(())
model.save(path.join(MODEL_DIR_PATH, MODEL_FILE_FMT.format(USR, model_name)))
print('Model saved as: ' + MODEL_FILE_FMT.format(USR, model_name))