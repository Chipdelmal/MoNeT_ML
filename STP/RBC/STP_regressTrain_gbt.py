
import STP_wrapper
from sys import argv
from sklearn.ensemble import GradientBoostingRegressor


(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[1], reg=True)

###############################################################################
# Training Model - Gradient Boosting Regressor
###############################################################################
clf = GradientBoostingRegressor(
    n_estimators=TREES,
    min_samples_split=5, min_samples_leaf=50,
    max_features=None, max_leaf_nodes=None
)
STP_wrapper.wrapperTrain(clf, 'GBT', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, reg=True)