import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

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

DATA_SCA = pd.get_dummies(DATA_SCA, columns=['i_sex'])
DATA_SCA = DATA_SCA.rename(columns={'i_sex_1': 'i_sxm', 'i_sex_2': 'i_sxg', 'i_sex_3': 'i_sxn'})

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
# Training Model - Voting Classifier
###############################################################################
rf = RandomForestClassifier(
    n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
    min_samples_split=5, min_samples_leaf=50,
    max_features=None, max_leaf_nodes=None,
    n_jobs=JOB
)
v_estimators = [('rf', rf), 
('lr', LogisticRegression(multi_class='multinomial', random_state = 1)),
('gnb', GaussianNB())
]
vc = VotingClassifier(estimators=v_estimators, voting='hard', n_jobs=JOB)
# K-fold training -------------------------------------------------------------
vc_kScores = cross_val_score(
    vc, TRN_X, TRN_Y.values.ravel(), 
    cv=int(KFOLD), 
    scoring=metrics.make_scorer(metrics.f1_score, average='weighted')
)
vc_outLabels = set(list(TRN_Y.values.ravel()))
# Final training --------------------------------------------------------------
vc.fit(TRN_X, TRN_Y.values.ravel())
vc_PRD_Y = vc.predict(VAL_X)
(vc_accuracy, vc_f1, vc_precision, vc_recall, vc_jaccard) = (
    metrics.accuracy_score(VAL_Y, vc_PRD_Y),
    metrics.f1_score(VAL_Y, vc_PRD_Y, average='weighted'),
    metrics.precision_score(VAL_Y, vc_PRD_Y, average='weighted'),
    metrics.recall_score(VAL_Y, vc_PRD_Y, average='weighted'),
    metrics.jaccard_score(VAL_Y, vc_PRD_Y, average='weighted')
)
vc_report = metrics.classification_report(VAL_Y, vc_PRD_Y)
vc_confusionMat = metrics.plot_confusion_matrix(
    vc, VAL_X, VAL_Y, 
    # display_labels=list(range(len(set(outputs[outputs.columns[0]])))),
    cmap=cm.Blues, normalize=None
)
plt.savefig('CLS_VC_' + MTR + '.jpg', dpi=300)
###############################################################################
# Statistics & Model Export
###############################################################################
# Voting Model ---------------------------------------------------------
# plt.savefig(modelPath+'_RF.jpg', dpi=300)
dump(vc, 'CLS_VC_' + MTR + '.joblib')
with open('CLS_VC_' + MTR + '.txt', 'w') as f:
    with redirect_stdout(f):
        print('* Voting Classifier')
        print('* Output Metric: ' + MTR)
        print('')
        print('* Feats Order: {}'.format(FEATS))
        print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
        print('* Cross-validation F1: %0.2f (+/-%0.2f)'%(vc_kScores.mean(), vc_kScores.std()*2))
        print('* Validation Accuracy: {:.2f}'.format(vc_accuracy))
        print('* Validation F1: {:.2f} ({:.2f}/{:.2f})'.format(vc_f1, vc_precision, vc_recall))
        print('* Jaccard: {:.2f}'.format(vc_jaccard))
        print('* Class report: ')
        print(vc_report) 