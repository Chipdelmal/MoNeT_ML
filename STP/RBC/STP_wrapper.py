import STP_constants as cst
from sys import argv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from joblib import dump
from contextlib import redirect_stdout
import rfpimp as rfp
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from os import path

def wrapperSetup(metric, dataset, path_arg):
    MTR = metric # 'CPT'
    (VT_SPLIT, TREES, DEPTH, KFOLD, JOB) = (
        cst.VT_SPLIT, cst.TREES, cst.DEPTH, cst.KFOLD, cst.JOB
    )

    ###############################################################################
    # Read CSV
    ###############################################################################
    if dataset == "REG":
        DATA = pd.read_csv(path.join(path_arg, 'REG_HLT_50Q_10T.csv'))
    elif dataset == "CLS": 
        DATA = pd.read_csv(path.join(path_arg, 'CLS_HLT_50Q_10T.csv'))
    elif dataset == "SCA":
        DATA = pd.read_csv(path.join(path_arg, 'A_SCA_HLT_50Q_10T.csv'))
    # Features and labels ---------------------------------------------------------
    COLS = list(DATA.columns)
    (FEATS, LABLS) = (
        [i for i in COLS if i[0]=='i'],
        [i for i in COLS if i[0]!='i']
    )

    ###############################################################################
    # Pre-Analysis
    ###############################################################################
    correlation = DATA.corr(method='spearman')
    (f, ax) = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation, mask=np.zeros_like(correlation, dtype=np.bool), 
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True, ax=ax
    )
    # f.show()

    ###############################################################################
    # Split Train/Test
    ###############################################################################
    (inputs, outputs) = (DATA[FEATS], DATA[MTR])
    
    #normalize features
    scaler = preprocessing.MinMaxScaler()
    inputs = pd.DataFrame(scaler.fit_transform(inputs), columns=FEATS)

    (TRN_X, VAL_X, TRN_Y, VAL_Y) = train_test_split(
        inputs, outputs, 
        test_size=float(VT_SPLIT)
    )
    (TRN_L, VAL_L) = [i.shape[0] for i in (TRN_X, VAL_X)]

    return [MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
    LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation]

def wrapperTrain(clf, model, TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, dataset, path_arg):
    #regression model
    if dataset == "REG" or dataset == "SCA":
        # K-fold training -------------------------------------------------------------
        kScores = cross_val_score(
            clf, TRN_X, TRN_Y, 
            cv=int(KFOLD), 
            scoring=metrics.make_scorer(metrics.r2_score)
        )
        #outLabels = set(list(TRN_Y))

        # Final training --------------------------------------------------------------
        clf.fit(TRN_X, TRN_Y)
        PRD_Y = clf.predict(VAL_X)

        # Evaluation --------------------------------------------------------------
        (r2, rmse, mae) = (
            metrics.r2_score(VAL_Y, PRD_Y),
            metrics.mean_squared_error(VAL_Y, PRD_Y, squared=False),
            metrics.mean_absolute_error(VAL_Y, PRD_Y)
        )

        ###############################################################################
        # Statistics & Model Export
        ###############################################################################
        dump(clf, path.join(path_arg, dataset + '_' + model + '_' + MTR + '.joblib'))
        with open(path.join(path_arg, dataset + '_' + model + '_' + MTR + '.txt'), 'w') as f:
            with redirect_stdout(f):
                print('* Output Metric: ' + MTR)
                print('')
                print('* Feats Order: {}'.format(FEATS))
                print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
                print('* Cross-validation R2: %0.2f (+/-%0.2f)'%(kScores.mean(), kScores.std()*2))
                print('* R2 score: {:.2f}'.format(r2))
                print('* Root Mean square error: {:.2f}'.format(rmse))
                print('* Mean absolute error: {:.2f}'.format(mae))
    #classifier model
    else:
        # K-fold training -------------------------------------------------------------
        kScores = cross_val_score(
            clf, TRN_X, TRN_Y.values.ravel(), 
            cv=int(KFOLD), 
            scoring=metrics.make_scorer(metrics.f1_score, average='weighted')
        )
        outLabels = set(list(TRN_Y.values.ravel()))
        # Final training --------------------------------------------------------------
        clf.fit(TRN_X, TRN_Y.values.ravel())
        PRD_Y = clf.predict(VAL_X)
        (accuracy, f1, precision, recall, jaccard) = (
            metrics.accuracy_score(VAL_Y, PRD_Y),
            metrics.f1_score(VAL_Y, PRD_Y, average='weighted'),
            metrics.precision_score(VAL_Y, PRD_Y, average='weighted'),
            metrics.recall_score(VAL_Y, PRD_Y, average='weighted'),
            metrics.jaccard_score(VAL_Y, PRD_Y, average='weighted')
        )
        report = metrics.classification_report(VAL_Y, PRD_Y)
        confusionMat = metrics.plot_confusion_matrix(
            clf, VAL_X, VAL_Y, 
            # display_labels=list(range(len(set(outputs[outputs.columns[0]])))),
            cmap=cm.Blues, normalize=None
        )
        plt.savefig(path.join(path_arg, dataset + '_' + model + '_' + MTR + '.png'), dpi=300)
        # Features importance ---------------------------------------------------------
        try:
            featImportance = list(clf.feature_importances_)
            impDC = rfp.oob_dropcol_importances(clf, TRN_X, TRN_Y.values.ravel())
            impDCD = impDC.to_dict()['Importance']
            impPM = rfp.importances(clf, TRN_X, TRN_Y)
            impPMD = impPM.to_dict()['Importance']
        except AttributeError:
            pass
        # viz = rfp.plot_corr_heatmap(DATA, figsize=(7,5))
        ###############################################################################
        # Statistics & Model Export
        ###############################################################################
        # plt.savefig(modelPath+'_RF.jpg', dpi=300)
        dump(clf, path.join(path_arg, dataset + '_' + model + '_' + MTR + '.joblib'))
        with open(path.join(path_arg, dataset + '_' + model + '_' + MTR + '.txt'), 'w') as f:
            with redirect_stdout(f):
                print('* Output Metric: ' + MTR)
                print('')
                print('* Feats Order: {}'.format(FEATS))
                print('* Train/Validate entries: {}/{} ({})'.format(TRN_L, VAL_L, TRN_L+VAL_L))
                print('* Cross-validation F1: %0.2f (+/-%0.2f)'%(kScores.mean(), kScores.std()*2))
                print('* Validation Accuracy: {:.2f}'.format(accuracy))
                print('* Validation F1: {:.2f} ({:.2f}/{:.2f})'.format(f1, precision, recall))
                print('* Jaccard: {:.2f}'.format(jaccard))
                if 'featImportance' in locals():
                    print('* Features Importance & Correlation')
                    for i in zip(FEATS, featImportance, correlation[LABLS[0]]):
                        print('\t* {}: {:.3f}, {:.3f}'.format(*i))
                    print('* Drop-Cols & Permutation Features Importance')
                    for i in FEATS:
                        print('\t* {}: {:.3f}, {:.3f}'.format(i, impDCD[i], impPMD[i]))
                print('* Class report: ')
                print(report)
