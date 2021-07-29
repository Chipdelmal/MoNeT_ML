
import STP_wrapper
from sys import argv
from sklearn.ensemble import BaggingRegressor, BaggingClassifier


(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[2], set=argv[1])

if argv[1] == 'SCA' or argv[1] == "REG":
    ###############################################################################
    # Training Model - Bagging Regressor
    ###############################################################################
    clf = BaggingRegressor(
        n_estimators=TREES
    )
else:
    ###############################################################################
    # Training Model - Bagging Classifier
    ###############################################################################
    clf = BaggingClassifier(
        n_estimators=TREES
    )

STP_wrapper.wrapperTrain(clf, 'B', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, set=argv[1])
