import STP_wrapper
from sys import argv
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
    LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[1], reg=False)

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
clf = VotingClassifier(estimators=v_estimators, voting='hard', n_jobs=JOB)

STP_wrapper.wrapperTrain(clf, 'VC', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, reg=False)