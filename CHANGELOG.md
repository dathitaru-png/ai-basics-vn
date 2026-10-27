# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/)
and semantic versioning.

## [0.2.0] - 2026-10-27

### Added
- Lesson 11: logistic regression — from regression to classification
  probabilities with from-scratch code (`logistic_regression.py`,
  sigmoid, gradient descent on binary cross-entropy)
- Lesson 12: evaluating classifiers beyond accuracy — confusion
  matrix, precision, recall and F1 with worked examples (`metrics.py`)
- Runnable examples `examples/demo_logistic.py` and
  `examples/demo_metrics.py`, plus `aibasics demo logistic` and
  `aibasics demo metrics` CLI demos
- 11 new tests (31 total); 4 new review-quiz questions, 1 new essay
  question, and 12 new EN–VI glossary entries for lessons 11–12

## [0.1.0] - 2026-10-23

### Added
- Vietnamese beginner curriculum: 10 lessons covering AI/ML/DL/GenAI
  concepts, data & evaluation, linear regression, k-NN, decision trees,
  Naive Bayes text classification, k-Means clustering, tiny neural
  networks (perceptron & MLP), LLM & prompting basics, and AI ethics
- EN–VI glossary (`lessons/thuat-ngu.md`) and a review quiz with answers
  (`lessons/cau-hoi-on-tap.md`)
- From-scratch, dependency-free implementations in `aibasics`:
  `LinearRegression` (gradient descent), `KNeighborsClassifier`,
  `DecisionTreeClassifier`, `MultinomialNaiveBayes`, `KMeans`
  (k-means++ seeding), `Perceptron`, `TinyMLP` (backpropagation)
- Synthetic (fictional) dataset helpers and metrics in `aibasics.data`
- CLI: `aibasics demo <name|all>` running every lesson demo
- Runnable examples in `examples/` mirroring lessons 03–08
- Pytest suite (20 tests), CI workflow on Python 3.10–3.12, and
  maintainer documentation
