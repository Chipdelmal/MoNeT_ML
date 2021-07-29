
import STP_wrapper
from sys import argv
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB

(MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation) = STP_wrapper.wrapperSetup(argv[2], set=argv[1])

if argv[1] == 'SCA' or argv[1] == 'REG':
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

else:
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

STP_wrapper.wrapperTrain(clf, 'VR', TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, set=argv[1])
