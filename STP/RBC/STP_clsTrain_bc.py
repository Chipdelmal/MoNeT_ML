import STP_wrapper
from sys import argv
from sklearn.ensemble import BaggingClassifier

(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
    LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[1], reg=False)
###############################################################################
# Training Model - Bagging Classifier
###############################################################################
clf = BaggingClassifier(
    n_estimators=TREES
)

STP_wrapper.wrapperTrain(clf, 'BC', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, reg=False)