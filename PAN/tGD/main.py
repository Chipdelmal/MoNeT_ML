from model import forest
from model import util
from model import svm

import sklearn.model_selection
import pandas as pd
import sys
import os

# Possible targets:  '0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95'

# Possible features:
# TGD: 'i_hnf', 'i_cac', 'i_frc', 'i_hrt', 'i_ren', 'i_res', 'i_key'
# PAN: 'i_smx', 'i_sgv', 'i_sgn', 'i_rsg', 'i_rer', 'i_ren', 'i_qnt', 'i_gsv', 'i_fic', 'i_key'

if __name__ == "__main__":

    # Edit Zone
    FEATURES    = ['i_smx', 'i_sgv', 'i_sgn', 'i_rsg', 'i_rer', 'i_ren', 'i_qnt', 'i_gsv', 'i_fic', 'i_key']
    TARGET      = ['0.5']
    KFOLDS      = 20
    OUTDIR      = 'output-svm-test'

    # Create output directory
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    # Collect data
    data = util.get_data('CLN_HLT_WOP', os.path.join('..', '..', '..', 'data'))
    
    # Format data
    matrices = util.stratified_split(
        data[FEATURES]
        , data[TARGET]
        , n_splits=1
        , test_size=0.3
        , random_state=42
    )

    print("Number of classes:", len(set(matrices[2].codes)))

    # Sample code for random forest classifier
    # model = forest.Classifier(
    #     matrices[0]
    #     , matrices[1]
    #     , matrices[2].codes
    #     , matrices[3].codes
    #     , n_estimators=30
    #     , max_depth=15
    #     , criterion='entropy'
    #     , min_samples_split=5
    #     , min_samples_leaf=50
    #     , max_features=None
    #     , max_leaf_nodes=None
    #     , random_state=42
    # )

    # Sample code for the support vector machine classifier
    model = svm.Classifier(
        matrices[0]
        , matrices[1]
        , matrices[2].codes
        , matrices[3].codes
        , kernel="sigmoid"
        , max_iter=100
        , random_state=42
    )

    # Send outputs to the output directory
    model.export_report(OUTDIR, 'report', folds=KFOLDS)
    model.export_model(OUTDIR, 'model')
    model.export_confusion_matrix(OUTDIR, 'matrix')
