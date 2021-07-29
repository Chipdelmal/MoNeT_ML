
from math import e
import STP_wrapper
from sys import argv
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, StackingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[2], set=argv[1])

if argv[1] == 'SCA' or argv[1] == 'REG':
    ###############################################################################
    # Training Model - Stacking Regressor
    ###############################################################################
    rf = RandomForestRegressor(
        n_estimators=TREES, max_depth=DEPTH,
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None,
        n_jobs=JOB
    )
    estimators = [
        ('svr', LinearSVR(random_state=42)),
        ('lr', RidgeCV())
    ]
    clf = StackingRegressor(
        estimators = estimators, final_estimator = rf,
        n_jobs = JOB
    )

else:
    ###############################################################################
    # Training Model - Stacking Classifier
    ###############################################################################
    rf = RandomForestClassifier(
        n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
        min_samples_split=5, min_samples_leaf=50,
        max_features=None, max_leaf_nodes=None,
        n_jobs=JOB
    )
    estimators = [('rf', rf), ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
    clf = StackingClassifier(
        estimators = estimators, n_jobs = JOB
    )

STP_wrapper.wrapperTrain(clf, 'SR', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, set=argv[1])