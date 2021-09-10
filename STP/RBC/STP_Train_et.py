import STP_wrapper
from sys import argv
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[2], argv[1], argv[3])

if argv[1] == 'SCA' or argv[1] == 'REG':
    ###############################################################################
    # Training Model - Extra Trees Regressor
    ###############################################################################
    clf = ExtraTreesRegressor(
        n_estimators=TREES, max_depth=DEPTH,
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None,
        n_jobs=JOB
    )

else:
    ###############################################################################
    # Training Model - Extra Trees Classifier
    ###############################################################################
    clf = ExtraTreesClassifier(
        n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None,
        n_jobs=JOB, bootstrap=True
    )

STP_wrapper.wrapperTrain(clf, 'ET', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, argv[1], argv[3])