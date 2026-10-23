# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/)
and semantic versioning.

## [Unreleased]

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
