# This is an example model implementation. The actual model architecture, hyperparameters can
# vary per use-case basis.

from sklearn.svm import SVC
import numpy as np

class SVCModel:
    def __init__(self,hparams):
        self.clf = SVC(hparams.C, hparams.kernel)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X,y)
    
    def predict(self,X: np.ndarray):
        return self.clf.predict(X)