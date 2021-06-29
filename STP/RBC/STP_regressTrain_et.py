import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn import metrics
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor

VT_SPLIT = .3
TREES = 100
DEPTH = 15
KFOLD = 20
MTR = "CPT"
# On Disk -> JOB = 8, On Server -> JOB = 60
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
    test_size=float(VT_SPLIT)
)
(TRN_L, VAL_L) = [i.shape[0] for i in (TRN_X, VAL_X)]

###############################################################################
# Training Model - Extra Trees Regressor
###############################################################################
clf = ExtraTreesRegressor(
    n_estimators=TREES, max_depth=DEPTH,
    min_samples_split=5, min_samples_leaf=50,
    max_features=None, max_leaf_nodes=None,
    n_jobs=JOB
)
# K-fold training -------------------------------------------------------------
kScores = cross_val_score(
    clf, TRN_X, TRN_Y, 
    cv=int(KFOLD), 
    scoring=metrics.make_scorer(metrics.r2_score)
)
#outLabels = set(list(TRN_Y))

# Final training --------------------------------------------------------------
clf.fit(TRN_X, TRN_Y)
PRD_Y = clf.predict(VAL_X)

# Evaluation --------------------------------------------------------------
(r2, rmse, mae) = (
    metrics.r2_score(VAL_Y, PRD_Y),
    metrics.mean_squared_error(VAL_Y, PRD_Y, squared=False),
    metrics.mean_absolute_error(VAL_Y, PRD_Y)
)

###############################################################################
# Statistics & Model Export
###############################################################################
dump(clf, 'REG_ET_' + MTR + '.joblib')
with open('REG_ET_' + MTR + '.txt', 'w') as f:
    with redirect_stdout(f):
        print('* Extra Trees Regressor')
        print('* Output Metric: ' + MTR)
        print('')
        print('* Feats Order: {}'.format(FEATS))
        print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
        print('* Cross-validation R2: %0.2f (+/-%0.2f)'%(kScores.mean(), kScores.std()*2))
        print('* R2 score: {:.2f}'.format(r2))
        print('* Root Mean square error: {:.2f}'.format(rmse))
        print('* Mean absolute error: {:.2f}'.format(mae))
