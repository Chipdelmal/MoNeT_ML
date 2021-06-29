import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier

VT_SPLIT = .3
TREES = 100
DEPTH = 15
KFOLD = 20
MTR = "POE"
# Running on Disk so JOB = 8
JOB = 8

###############################################################################
# Read CSV
###############################################################################
DATA_SCA = pd.read_csv('SCA_HLT_50Q_10T.csv')
DATA_REG = pd.read_csv('REG_HLT_50Q_10T.csv')
DATA_CLS = pd.read_csv('CLS_HLT_50Q_10T.csv')

frames = [DATA_SCA, DATA_REG, DATA_CLS]
DATA = pd.concat(frames)
# Features and labels ---------------------------------------------------------
COLS = list(DATA.columns)
(FEATS, LABLS) = (
    [i for i in COLS if i[0]=='i'],
    [i for i in COLS if i[0]!='i']
)
###############################################################################
# Pre-Analysis
###############################################################################
correlation = DATA.corr(method='spearman')
(f, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation, mask=np.zeros_like(correlation, dtype=np.bool), 
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    square=True, ax=ax
)
# f.show()
###############################################################################
# Split Train/Test
###############################################################################
(inputs, outputs) = (DATA[FEATS], DATA[MTR])
(TRN_X, VAL_X, TRN_Y, VAL_Y) = train_test_split(
    inputs, outputs, 
    test_size=float(VT_SPLIT),
    stratify=outputs
)
(TRN_L, VAL_L) = [i.shape[0] for i in (TRN_X, VAL_X)]
###############################################################################
# Training Model - Bagging Classifier
###############################################################################
bc = BaggingClassifier(
    n_estimators=TREES
)
# K-fold training -------------------------------------------------------------
bc_kScores = cross_val_score(
    bc, TRN_X, TRN_Y.values.ravel(), 
    cv=int(KFOLD), 
    scoring=metrics.make_scorer(metrics.f1_score, average='weighted')
)
bc_outLabels = set(list(TRN_Y.values.ravel()))
# Final training --------------------------------------------------------------
bc.fit(TRN_X, TRN_Y.values.ravel())
bc_PRD_Y = bc.predict(VAL_X)
(bc_accuracy, bc_f1, bc_precision, bc_recall, bc_jaccard) = (
    metrics.accuracy_score(VAL_Y, bc_PRD_Y),
    metrics.f1_score(VAL_Y, bc_PRD_Y, average='weighted'),
    metrics.precision_score(VAL_Y, bc_PRD_Y, average='weighted'),
    metrics.recall_score(VAL_Y, bc_PRD_Y, average='weighted'),
    metrics.jaccard_score(VAL_Y, bc_PRD_Y, average='weighted')
)
bc_report = metrics.classification_report(VAL_Y, bc_PRD_Y)
bc_confusionMat = metrics.plot_confusion_matrix(
    bc, VAL_X, VAL_Y, 
    # display_labels=list(range(len(set(outputs[outputs.columns[0]])))),
    cmap=cm.Blues, normalize=None
)
plt.savefig('CLS_BC_' + MTR + '.jpg', dpi=300)
###############################################################################
# Statistics & Model Export
###############################################################################
# Bagging Model ---------------------------------------------------------
# plt.savefig(modelPath+'_RF.jpg', dpi=300)
dump(bc, 'CLS_BC_' + MTR + '.joblib')
with open('CLS_BC_' + MTR + '.txt', 'w') as f:
    with redirect_stdout(f):
        print('* Feats Order: {}'.format(FEATS))
        print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
        print('* Cross-validation F1: %0.2f (+/-%0.2f)'%(bc_kScores.mean(), bc_kScores.std()*2))
        print('* Validation Accuracy: {:.2f}'.format(bc_accuracy))
        print('* Validation F1: {:.2f} ({:.2f}/{:.2f})'.format(bc_f1, bc_precision, bc_recall))
        print('* Jaccard: {:.2f}'.format(bc_jaccard))
        print('* Class report: ')
        print(bc_report)