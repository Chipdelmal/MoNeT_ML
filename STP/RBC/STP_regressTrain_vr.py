
import STP_wrapper
from sys import argv
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[1], reg=True)

###############################################################################
# Training Model - Voting Regressor
###############################################################################
rf = RandomForestRegressor(
    n_estimators=TREES, max_depth=DEPTH,
    min_samples_split=5, min_samples_leaf=50,
    max_features=None, max_leaf_nodes=None,
    n_jobs=JOB
)
estimators = [('lr', LinearRegression()), ('rf', rf)]
clf = VotingRegressor(estimators=estimators, n_jobs=JOB)

STP_wrapper.wrapperTrain(clf, 'VR', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, reg=True)
