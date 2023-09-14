# This is an example model implementation. The actual model architecture, hyperparameters can
# vary per use-case basis.

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import logging
import numpy as np

class SVCModel:
    def __init__(self):

        self.clf = make_pipeline(StandardScaler(), SVC())
        self.has_been_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        logging.info(f"SVCModel.fit called on {X.shape=} and {y.shape=}")
        if self.has_been_fit:
            logging.warn(f"calling fit on an already fitted {self.__class__.__name__} object.")

        self.clf.fit(X,y)
        self.has_been_fit = True
        logging.info(f"SVCModel.fit called on {X.shape=} and {y.shape=}")
    
    def predict(self, X: np.ndarray):

        if not self.has_been_fit:
            raise ValueError(f"calling predict on {self.__class__.__name__} object without calling fit first.")

        return self.clf.predict(X)