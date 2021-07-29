
import STP_wrapper
from sys import argv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[2], set=argv[1])

if argv[1] == 'SCA' or argv[1] == "REG":
    ###############################################################################
    # Training Model - Random Forest Regressor
    ###############################################################################
    clf = RandomForestRegressor(
        n_estimators=TREES, max_depth=DEPTH,
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None,
        n_jobs=JOB
    )

else:
    ###############################################################################
    # Training Model - Random Forest Classifier
    ###############################################################################
    clf = RandomForestClassifier(
        n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None,
        n_jobs=JOB, bootstrap=True
    )

STP_wrapper.wrapperTrain(clf, 'RF', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, set=argv[1])
