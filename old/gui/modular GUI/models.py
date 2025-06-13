"""
models.py – scikit‑learn wrappers.  Signature: model_func(X, y=None, **params)
"""
from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def _wrap(clf):
    return clf, (lambda X, y=None: np.nan if y is None else clf.score(X, y))

def lda(X, y=None, **p):    return _wrap(LinearDiscriminantAnalysis().fit(X, y) if y is not None else LinearDiscriminantAnalysis())
def logreg(X, y=None, **p): return _wrap(LogisticRegression(max_iter=1000).fit(X, y) if y is not None else LogisticRegression())
def svm(X, y=None, **p):    return _wrap(SVC(kernel="rbf", probability=True).fit(X, y) if y is not None else SVC())

MODEL_FUNCS = {"LDA": lda, "LogReg": logreg, "SVM": svm,
               "CNN": lambda X, y=None, **p: ("cnn", lambda *_: 0.0),
               "RNN": lambda X, y=None, **p: ("rnn", lambda *_: 0.0),
               "None": lambda X, y=None, **p: ("dummy", lambda *_: 0)}
