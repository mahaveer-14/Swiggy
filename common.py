"""Shared helpers for Parts 2 and 4 (importing this file never runs a Part).
Standard library + NumPy only; every statistic here is computed by hand-written code."""
import csv
import math
import os

import numpy as np


# ------------------------------------------------------------------ results formatting
class Results:
    """Collects `key = value` lines in the format the brief requires."""

    def __init__(self):
        self.lines = []

    def put(self, key, value):
        if isinstance(value, (bool, np.bool_)):
            text = "yes" if value else "no"
        elif isinstance(value, (int, np.integer)):
            text = str(int(value))
        elif isinstance(value, str):
            text = value
        else:
            text = format(float(value), ".4f")
        self.lines.append((key, text))

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for key, text in self.lines:
                f.write(f"{key} = {text}\n")


def f4(x):
    """Float printed with 4 decimals (for CSV files)."""
    return format(float(x), ".4f")


# ------------------------------------------------------------------ data
def load_records(path=os.path.join("data", "records.csv")):
    """Return X (800 x 7: X1..X7 in file order), outcome (0/1 ints) and value."""
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    X = np.array([[float(v) for v in r[1:8]] for r in rows])
    outcome = np.array([int(float(r[8])) for r in rows])
    value = np.array([float(r[9]) for r in rows])
    return X, outcome, value


def stratified_split(outcome):
    """Task 2.1: per class, the first floor(7 * count / 10) rows (file order) are training."""
    train = []
    for cls in (0, 1):
        idx = np.where(outcome == cls)[0]
        k = (7 * len(idx)) // 10          # integer arithmetic
        train.extend(idx[:k].tolist())
    train = np.array(sorted(train))        # both sets keep file order
    test = np.array(sorted(set(range(len(outcome))) - set(train.tolist())))
    return train, test


def sample_sd_cols(A):
    """Column-wise sample SD (divisor n - 1)."""
    d = A - A.mean(axis=0)
    return np.sqrt((d * d).sum(axis=0) / (A.shape[0] - 1))


def standardise(train_X, other_X):
    """Standardise with the TRAINING mean and SD only; apply the same numbers to other rows."""
    mu = train_X.mean(axis=0)
    sd = sample_sd_cols(train_X)
    return (train_X - mu) / sd, (other_X - mu) / sd


# ------------------------------------------------------------------ metrics
def confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def _ratio(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    """accuracy, precision, recall, f1, specificity, fpr (0 when a denominator is 0)."""
    accuracy = _ratio(tp + tn, tp + fp + tn + fn)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    specificity = _ratio(tn, tn + fp)
    fpr = _ratio(fp, fp + tn)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "specificity": specificity, "fpr": fpr}


def cost_money(fp, fn):
    """Task 2.8: INR 50 per false alarm, INR 400 per missed positive."""
    return 50 * fp + 400 * fn


# ------------------------------------------------------------------ logistic regression
def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    """Binary cross-entropy averaged over rows."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def train_logistic(X, y, eta, updates, record=()):
    """Batch gradient descent from zero. Returns (w, b, losses) where losses maps an update
    count (0 and every value in `record`) to the training loss after that many updates."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = {0: bce(y, predict_proba(X, w, b))}
    for step in range(1, updates + 1):
        p = predict_proba(X, w, b)
        err = p - y
        grad_w = (X.T @ err) / n
        grad_b = float(err.sum()) / n
        w = w - eta * grad_w
        b = b - eta * grad_b
        if step in record:
            losses[step] = bce(y, predict_proba(X, w, b))
    return w, b, losses


def prepare_split():
    """Task 2.1 in one call. Returns (Xtr_raw, Xte_raw, Xtr, Xte, ytr, yte): raw columns X1..X7 in
    original units and the same rows standardised with the training mean / SD, for train and test."""
    X, outcome, _value = load_records()
    train_idx, test_idx = stratified_split(outcome)
    Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
    Xtr, Xte = standardise(Xtr_raw, Xte_raw)
    return Xtr_raw, Xte_raw, Xtr, Xte, outcome[train_idx], outcome[test_idx]


# ------------------------------------------------------------------ k-nearest neighbours
def knn_positive_counts(ref_X, ref_y, query_X, ks):
    """For each K in `ks`, the number of positive neighbours among the K nearest reference rows.
    Neighbours are ranked by (Euclidean distance, then reference position in file order);
    a stable sort keeps that order. Returns {K: array of counts per query row}."""
    diff = query_X[:, None, :] - ref_X[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    order = np.argsort(dist, axis=1, kind="stable")
    labels = ref_y[order]                    # labels of neighbours, nearest first
    cum = np.cumsum(labels, axis=1)
    return {k: cum[:, k - 1] for k in ks}


def knn_predict(counts, k):
    """Positive when at least (K + 1) / 2 of the K neighbours are positive (K odd)."""
    return (counts >= (k + 1) // 2).astype(int)


KNN_KS = list(range(1, 22, 2))                  # K = 1, 3, ..., 21


def knn_cv_scores(Xtr, ytr, ks=KNN_KS):
    """Task 2.7: 5-fold CV on the training rows (fold = position in the training set mod 5).
    The features are already standardised with the training mean / SD and are NOT re-standardised
    inside a fold. Returns {K: mean over the five folds of the held-out F1}."""
    fold = np.arange(len(ytr)) % 5
    fold_f1 = {k: [] for k in ks}
    for fo in range(5):
        ref = fold != fo
        qry = fold == fo
        counts = knn_positive_counts(Xtr[ref], ytr[ref], Xtr[qry], ks)
        for k in ks:
            tp, fp, tn, fn = confusion(ytr[qry], knn_predict(counts[k], k))
            fold_f1[k].append(metrics(tp, fp, tn, fn)["f1"])
    return {k: sum(fold_f1[k]) / 5.0 for k in ks}


def choose_best_k(mean_cv):
    """Highest mean F1; ties go to the smaller K (ascending order with a strict >)."""
    ks = sorted(mean_cv)
    best = ks[0]
    for k in ks:
        if mean_cv[k] > mean_cv[best]:
            best = k
    return best
