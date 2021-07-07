
import STP_wrapper
from sys import argv
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import RidgeCV


(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[1], reg=True)

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

STP_wrapper.wrapperTrain(clf, 'SR', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, reg=True)