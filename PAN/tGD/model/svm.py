from model.classifier import BaseModel
from sklearn import svm

class Classifier(BaseModel):

    def __init__(self, X_train, X_test, y_train, y_test, *model_args, **model_kwargs):
        super().__init__(svm.SVC
          , X_train
          , X_test
          , y_train
          , y_test
          , *model_args
          , **model_kwargs
        )
