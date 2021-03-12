from model import forest
from model import util

import sklearn.model_selection
import pandas as pd
import sys
import os

# Possible features: 'i_hnf', 'i_cac', 'i_frc', 'i_hrt', 'i_ren', 'i_res', 'i_key'
# Possible targets:  '0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95'

if __name__ == "__main__":

    FEATURES    = ['i_hnf', 'i_ren', 'i_key']
    TARGET      = ['0.5']
    KFOLDS      = 20
    CLASSIFY    = True
    OUT_DIR     = 'out'

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    data = util.get_data('WOP', os.path.join('..', 'data'))
    
    if CLASSIFY:
        matrices = util.stratified_split(
            data[FEATURES]
            , data[TARGET]
            , n_splits=1
            , test_size=0.3
            , random_state=42
        )
        rand_fst = forest.Classifier(
            matrices[0]
            , matrices[1]
            , matrices[2].codes
            , matrices[3].codes
            , n_estimators=30
            , max_depth=15
            , criterion='entropy'
            , min_samples_split=5
            , min_samples_leaf=50
            , max_features=None
            , max_leaf_nodes=None
            , random_state=42
        )
        rand_fst.export_report(OUT_DIR, 'report', folds=KFOLDS)
        rand_fst.export_model(OUT_DIR, 'model')
        rand_fst.export_confusion_matrix(OUT_DIR, 'matrix')
    else:
        matrices = sklearn.model_selection.train_test_split(
            data[FEATURES]
            , data[TARGET]
            , test_size=0.3
            , random_state=42
        )
        rand_fst = forest.Regressor(*matrices, random_state=42)
        print(rand_fst.compute(KFOLDS))
