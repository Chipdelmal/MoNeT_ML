
import STP_wrapper
from sys import argv
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[2], argv[1], argv[3])

if argv[1] == 'SCA' or argv[1] == 'REG':
    ###############################################################################
    # Training Model - Gradient Boosting Regressor
    ###############################################################################
    clf = GradientBoostingRegressor(
        n_estimators=TREES,
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None
    )

else:
    ###############################################################################
    # Training Model - Gradient Boosted Trees
    ###############################################################################
    clf = GradientBoostingClassifier(
        n_estimators=TREES, max_depth=DEPTH,
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None
    )
STP_wrapper.wrapperTrain(clf, 'GBT', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, argv[1], argv[3])