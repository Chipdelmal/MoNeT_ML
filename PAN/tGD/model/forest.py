from sklearn import model_selection
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import contextlib
import joblib
import json
import os

class AbstractRandomForestModel:

    pipeline = pipeline.Pipeline([
        ('std_scaler', preprocessing.StandardScaler())
    ])

    def __init__(self, tree, X_train, X_test, y_train, y_test):
        self._tree = tree
        self._cols = X_train.columns
        self._corr = pd.concat([
            pd.DataFrame(np.c_[X_train.values, y_train], columns=list(self._cols.values) + ['target']),
            pd.DataFrame(np.c_[X_test.values , y_test ], columns=list(self._cols.values) + ['target'])
        ], ignore_index=True).corr(method='spearman').loc[self._cols, 'target']
        self._X_tr = self.pipeline.fit_transform(X_train)
        self._X_te = self.pipeline.fit_transform(X_test)
        self._y_tr = y_train
        self._y_te = y_test

    def compute(self, *args, **kwargs):
        raise NotImplementedError()

    def export_report(self, dirname, fname, width=30, *args, **kwargs):
        with open(os.path.join(dirname, fname + '.txt'), 'w') as f:
            with contextlib.redirect_stdout(f):
                for k, v in self.compute(*args, **kwargs).items():
                    print("{key:{width}}:".format(key=k, width=width), v)

    def export_model(self, dirname, fname):
        if self._tree is not None:
            joblib.dump(self._tree, os.path.join(dirname, fname + '.joblib'))

class Classifier(AbstractRandomForestModel):

    def __init__(self, X_train, X_test, y_train, y_test, *model_args, **model_kwargs):
        super().__init__(ensemble.RandomForestClassifier(*model_args, **model_kwargs)
            , X_train
            , X_test
            , y_train
            , y_test
        )

    def compute(self, folds):
        self._tree.fit(self._X_tr, self._y_tr)
        te_predictions = self._tree.predict(self._X_te)
        tr_predictions = self._tree.predict(self._X_tr)
        kscores = model_selection.cross_val_score(self._tree
            , self._X_tr
            , self._y_tr
            , cv=folds
            , scoring=metrics.make_scorer(metrics.f1_score, average='weighted')
        )
        return {
            "Dataset size"                  : len(self._X_tr) + len(self._X_te),
            "Training size"                 : len(self._X_tr),
            "Testing size"                  : len(self._X_te),
            "Training set accuracy score"   : metrics.accuracy_score    (self._y_tr, tr_predictions),
            "Training set F1 score"         : metrics.f1_score          (self._y_tr, tr_predictions, average="weighted"),
            "Training set precision score"  : metrics.precision_score   (self._y_tr, tr_predictions, average="weighted"),
            "Training set recall score"     : metrics.precision_score   (self._y_tr, tr_predictions, average="weighted"),
            "Training set Jaccard score"    : metrics.jaccard_score     (self._y_tr, tr_predictions, average="weighted"),
            "Test set accuracy score"       : metrics.accuracy_score    (self._y_te, te_predictions),
            "Test set F1 score"             : metrics.f1_score          (self._y_te, te_predictions, average="weighted"),
            "Test set precision score"      : metrics.precision_score   (self._y_te, te_predictions, average="weighted"),
            "Test set recall score"         : metrics.precision_score   (self._y_te, te_predictions, average="weighted"),
            "Test set Jaccard score"        : metrics.jaccard_score     (self._y_te, te_predictions, average="weighted"),
            "CV (f1) mean"                  : kscores.mean(),
            "CV (f1) stddev"                : kscores.std() * 2,
            "Feature importances"           : dict(zip(self._cols, self._tree.feature_importances_)),
            "Correlations"                  : dict(zip(self._cols, self._corr)),
            "Report"                        : "\n" + metrics.classification_report(self._y_te, te_predictions, zero_division=0)
        }
    
    def export_confusion_matrix(self, dirname, fname):
        mtx = metrics.plot_confusion_matrix(
            self._tree
            , self._X_te
            , self._y_te
            , display_labels=[
                'None'
                , 'Low'
                , 'Mid'
                , 'High'
                , 'Permanent'
            ]
            , cmap=plt.cm.Blues
            , normalize=None
        )
        plt.savefig(os.path.join(dirname, fname + '.jpg'), dpi=300)

class Regressor(AbstractRandomForestModel):

    def __init__(self, X_train, X_test, y_train, y_test, *model_args, **model_kwargs):
        super().__init__(ensemble.RandomForestRegressor(*model_args, **model_kwargs)
            , X_train
            , X_test
            , y_train.values.ravel()
            , y_test.values.ravel()
        )

    def compute(self, folds):
        self._tree.fit(self._X_tr, self._y_tr)
        kscores = model_selection.cross_val_score(self._tree
            , self._X_tr
            , self._y_tr
            , cv=folds
            , scoring=metrics.make_scorer(metrics.mean_squared_error)
        )
        return {
            "Dataset size"          : len(self._X_tr) + len(self._X_te),
            "Training size"         : len(self._X_tr),
            "Testing size"          : len(self._X_te),
            "Training error"        : metrics.mean_squared_error(self._y_tr, self._tree.predict(self._X_tr)),
            "Testing error"         : metrics.mean_squared_error(self._y_te, self._tree.predict(self._X_te)),
            "CV (f1) mean"          : kscores.mean(),
            "CV (f1) stddev"        : kscores.std() * 2,
            "Feature importances"   : dict(zip(self._cols, self._tree.feature_importances_)),
            "Correlations"          : dict(zip(self._cols, self._corr))
        }
        