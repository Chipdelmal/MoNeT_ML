import numpy as np
import pandas as pd
import rfpimp as rfp
import seaborn as sns
from joblib import dump
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

VT_SPLIT = .3
TREES = 100
DEPTH = 15
KFOLD = 20
MTR = "CPT"
# Running on Disk so JOB = 8
JOB = 8

###############################################################################
# Read CSV
###############################################################################
DATA_SCA = pd.read_csv('SCA_HLT_50Q_10T.csv')
"""
DATA_REG = pd.read_csv('REG_HLT_50Q_10T.csv')
DATA_CLS = pd.read_csv('CLS_HLT_50Q_10T.csv')

DATA_SCA = pd.get_dummies(DATA_SCA, columns=['i_sex'])
DATA_SCA = DATA_SCA.rename(columns={'i_sex_1': 'i_sxm', 'i_sex_2': 'i_sxg', 'i_sex_3': 'i_sxn'})

frames = [DATA_SCA, DATA_REG, DATA_CLS]
DATA = pd.concat(frames)
"""
DATA = DATA_SCA

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
# Training Model - Random Forest
###############################################################################
rf = RandomForestClassifier(
    n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
    min_samples_split=5, min_samples_leaf=50,
    max_features=None, max_leaf_nodes=None,
    n_jobs=JOB
)
# K-fold training -------------------------------------------------------------
kScores = cross_val_score(
    rf, TRN_X, TRN_Y.values.ravel(), 
    cv=int(KFOLD), 
    scoring=metrics.make_scorer(metrics.f1_score, average='weighted')
)
outLabels = set(list(TRN_Y.values.ravel()))
# Final training --------------------------------------------------------------
rf.fit(TRN_X, TRN_Y.values.ravel())
PRD_Y = rf.predict(VAL_X)
(accuracy, f1, precision, recall, jaccard) = (
    metrics.accuracy_score(VAL_Y, PRD_Y),
    metrics.f1_score(VAL_Y, PRD_Y, average='weighted'),
    metrics.precision_score(VAL_Y, PRD_Y, average='weighted'),
    metrics.recall_score(VAL_Y, PRD_Y, average='weighted'),
    metrics.jaccard_score(VAL_Y, PRD_Y, average='weighted')
)
report = metrics.classification_report(VAL_Y, PRD_Y)
confusionMat = metrics.plot_confusion_matrix(
    rf, VAL_X, VAL_Y, 
    # display_labels=list(range(len(set(outputs[outputs.columns[0]])))),
    cmap=cm.Blues, normalize=None
)
plt.savefig('CLS_RF_' + MTR + '.jpg', dpi=300)
# Features importance ---------------------------------------------------------
featImportance = list(rf.feature_importances_)
impDC = rfp.oob_dropcol_importances(rf, TRN_X, TRN_Y.values.ravel())
impDCD = impDC.to_dict()['Importance']
impPM = rfp.importances(rf, TRN_X, TRN_Y)
impPMD = impPM.to_dict()['Importance']
# viz = rfp.plot_corr_heatmap(DATA, figsize=(7,5))
###############################################################################
# Statistics & Model Export
###############################################################################
# Random Forest Model ---------------------------------------------------------
# plt.savefig(modelPath+'_RF.jpg', dpi=300)
dump(rf, 'CLS_RF_' + MTR + '.joblib')
with open('CLS_RF_' + MTR + '.txt', 'w') as f:
    with redirect_stdout(f):
        print('* Random Forest Classifier')
        print('* Output Metric: ' + MTR)
        print('')
        print('* Feats Order: {}'.format(FEATS))
        print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
        print('* Cross-validation F1: %0.2f (+/-%0.2f)'%(kScores.mean(), kScores.std()*2))
        print('* Validation Accuracy: {:.2f}'.format(accuracy))
        print('* Validation F1: {:.2f} ({:.2f}/{:.2f})'.format(f1, precision, recall))
        print('* Jaccard: {:.2f}'.format(jaccard))
        print('* Features Importance & Correlation')
        for i in zip(FEATS, featImportance, correlation[LABLS[0]]):
            print('\t* {}: {:.3f}, {:.3f}'.format(*i))
        print('* Drop-Cols & Permutation Features Importance')
        for i in FEATS:
            print('\t* {}: {:.3f}, {:.3f}'.format(i, impDCD[i], impPMD[i]))
        print('* Class report: ')
        print(report)