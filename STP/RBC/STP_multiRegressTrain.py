import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn import metrics
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

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
(inputs, outputs) = (DATA[FEATS], DATA[LABLS])
(TRN_X, VAL_X, TRN_Y, VAL_Y) = train_test_split(
    inputs, outputs, 
    test_size=float(VT_SPLIT)
)
(TRN_L, VAL_L) = [i.shape[0] for i in (TRN_X, VAL_X)]
###############################################################################
# Training Model - Multioutput Regressor using Gradient Boosting
###############################################################################
clf = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
print("starting k-fold training")
# K-fold training -------------------------------------------------------------
kScores = cross_val_score(
    clf, TRN_X, TRN_Y, 
    cv=int(KFOLD), 
    scoring=metrics.make_scorer(metrics.r2_score, multioutput='uniform_average')
)
#outLabels = set(list(TRN_Y))
print("past K-fold training")
# Final training --------------------------------------------------------------
clf.fit(TRN_X, TRN_Y)
print("past fitting")
PRD_Y = clf.predict(VAL_X)
print("past prediction")
(r2, rmse, mae) = (
    metrics.r2_score(VAL_Y, PRD_Y, multioutput='raw_values'),
    metrics.mean_squared_error(VAL_Y, PRD_Y, squared=False, multioutput='raw_values'),
    metrics.mean_absolute_error(VAL_Y, PRD_Y, multioutput='raw_values')
)
###############################################################################
# Statistics & Model Export
###############################################################################
dump(clf, 'REG_MULTI_' + MTR + '.joblib')
with open('REG_MULTI_' + MTR + '.txt', 'w') as f:
    with redirect_stdout(f):
        print('* Multioutput Regressor using Gradient Boosting')
        print('* Output Metric: ' + LABLS)
        print('')
        print('* Feats Order: {}'.format(FEATS))
        print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
        print('* Cross-validation R2: %0.2f (+/-%0.2f)'%(kScores.mean(), kScores.std()*2))
        print('* R2 score: '+ ''.join(['{} = {:.2f}, '.format(row[0], row[1]) for row in [[LABLS[i], r2[i]] for i in range(len(LABLS))]]))
        print('* Root Mean square error: '+ ''.join(['{} = {:.2f}, '.format(row[0], row[1]) for row in [[LABLS[i], rmse[i]] for i in range(len(LABLS))]]))
        print('* Mean absolute error: '+ ''.join(['{} = {:.2f}, '.format(row[0], row[1]) for row in [[LABLS[i], mae[i]] for i in range(len(LABLS))]]))
        