# This is an example model implementation. The actual model architecture, hyperparameters can
# vary per use-case basis.

import logging
from sklearn.svm import SVC
import numpy as np

class SVCModel:
    def __init__(self,hparams):

        self.clf = SVC(hparams.C, hparams.kernel)
        self.has_been_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray):

        if self.has_been_fit:
            logging.warn(f"calling fit on an already fitted {self.__class__.__name__} object.")

        self.clf.fit(X,y)
    
    def predict(self,X: np.ndarray):

        if not self.has_been_fit:
            raise ValueError(f"calling predict on {self.__class__.__name__} object without calling fit first.")

        return self.clf.predict(X)