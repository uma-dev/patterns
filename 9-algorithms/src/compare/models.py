from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def get_models() -> Dict[str, object]:
    """
    Classifiers commonly seen in class (simple defaults).
    """
    return {
        "LogReg": LogisticRegression(max_iter=2000),
        "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GaussianNB": GaussianNB(),
        "SVM(RBF)": SVC(),
    }
