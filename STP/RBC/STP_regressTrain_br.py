
import STP_constants as cst
from sys import argv
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn import metrics
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingRegressor


# Launch with: python STP_regressTrain_br.py 'CPT'

MTR = argv[1] # 'CPT'
(VT_SPLIT, TREES, DEPTH, KFOLD) = (
    cst.VT_SPLIT, cst.TREES, cst.DEPTH, cst.KFOLD
)

###############################################################################
# Read CSV
###############################################################################
DATA = pd.read_csv('REG_HLT_50Q_10T.csv')
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
# Training Model - Bagging Regressor
###############################################################################
clf = BaggingRegressor(
    n_estimators=TREES
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
dump(clf, 'REG_BR_' + MTR + '.joblib')
with open('REG_BR_' + MTR + '.txt', 'w') as f:
    with redirect_stdout(f):
        print('* Bagging Regressor')
        print('* Output Metric: ' + MTR)
        print('')
        print('* Feats Order: {}'.format(FEATS))
        print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
        print('* Cross-validation R2: %0.2f (+/-%0.2f)'%(kScores.mean(), kScores.std()*2))
        print('* R2 score: {:.2f}'.format(r2))
        print('* Root Mean square error: {:.2f}'.format(rmse))
        print('* Mean absolute error: {:.2f}'.format(mae))
