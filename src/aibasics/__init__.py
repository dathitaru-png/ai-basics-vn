"""AI Basics VN — basic AI knowledge for beginners, in Vietnamese.

An open learning kit: a structured beginner curriculum in Vietnamese
(``lessons/`` in the repository) plus small, readable, dependency-free
implementations of classic machine-learning algorithms, written from
scratch so any beginner can run them with just Python:

- :class:`LinearRegression` — gradient descent (lesson 03)
- :class:`KNeighborsClassifier` — k-NN (lesson 04)
- :class:`DecisionTreeClassifier` — decision tree (lesson 05)
- :class:`MultinomialNaiveBayes` — text classification (lesson 06)
- :class:`KMeans` — clustering (lesson 07)
- :class:`Perceptron` and :class:`TinyMLP` — tiny neural nets (lesson 08)
- :class:`LogisticRegression` — probability prediction (lesson 11)
- :mod:`aibasics.metrics` — confusion matrix, precision, recall, F1
  (lesson 12)
"""

from .data import (
    accuracy_score,
    make_blobs,
    make_regression_data,
    make_text_dataset,
    mean_squared_error,
    train_test_split,
)
from .decision_tree import DecisionTreeClassifier
from .kmeans import KMeans
from .knn import KNeighborsClassifier
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import MultinomialNaiveBayes, tokenize
from .perceptron import Perceptron, TinyMLP

__version__ = "0.1.0"

__all__ = [
    "DecisionTreeClassifier",
    "KMeans",
    "KNeighborsClassifier",
    "LinearRegression",
    "LogisticRegression",
    "MultinomialNaiveBayes",
    "Perceptron",
    "TinyMLP",
    "accuracy_score",
    "make_blobs",
    "make_regression_data",
    "make_text_dataset",
    "mean_squared_error",
    "tokenize",
    "train_test_split",
]
