"""Shared numerical engine for the Flipkart E-commerce Decision Intelligence capstone.

Every graded quantity in this project is produced by the code in this file and in
``part1.py`` .. ``part4.py``.  Only the Python standard library and NumPy are used:
the split rule, the standardisation, logistic regression, k-nearest neighbours,
k-means, the hierarchical linkages and PCA are all implemented from first
principles.  ``numpy.linalg`` is used only for the two places the brief allows it
(the normal-equation solve in Part 1 and the eigen-decomposition in Part 3).

Importing this module has no side effects: it never writes a file and never runs
a Part.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "records.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# --------------------------------------------------------------------------------------
# Column layout (fixed by the brief)
# --------------------------------------------------------------------------------------
COLUMNS = [
    "record_id",
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag",
    "churned",
    "order_value_inr_hundreds",
]

# X1..X7: the seven predictors, in file order.
FEATURE_NAMES = COLUMNS[1:8]
CONTINUOUS_5 = COLUMNS[1:6]  # X1..X5, used by Part 3

EXPECTED_FINGERPRINT = (800, 10, 170, "c6f249ba693f1ebeed318d6d5e18de23")

# Split constants
TRAIN_FRACTION_NUM = 7
TRAIN_FRACTION_DEN = 10

# Cost model (INR) used by Part 2 and Part 4
COST_FALSE_ALARM = 50.0
COST_MISSED_CHURN = 400.0


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------
def fmt(value) -> str:
    """Render one results value using the format pinned by the brief.

    Integers print as integers, booleans as ``yes``/``no``, floats with four
    decimal places, everything else as its plain text form.
    """
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


class Results:
    """An ordered ``key = value`` results file."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, object]] = []

    def add(self, key: str, value) -> None:
        self._rows.append((key, value))

    def add_many(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.add(key, value)

    def get(self, key: str):
        for k, v in self._rows:
            if k == key:
                return v
        raise KeyError(key)

    def write(self, filename: str) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", newline="") as handle:
            for key, value in self._rows:
                handle.write(f"{key} = {fmt(value)}\n")
        return path


def write_csv(filename: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Write a CSV with the pinned four-decimal float formatting."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([fmt_cell(cell) for cell in row])
    return path


def fmt_cell(value) -> str:
    """CSV cell formatting: floats to four decimals, everything else as text."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_records():
    """Load ``data/records.csv`` and return ``(ids, X, y, value)``.

    ``X`` has shape (800, 7) holding X1..X7 in file order, ``y`` is the churn
    outcome and ``value`` the order-value column.
    """
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"{DATA_PATH} is missing. Run `python generate_data.py` first."
        )

    ids: list[str] = []
    features: list[list[float]] = []
    outcome: list[int] = []
    value: list[float] = []

    with open(DATA_PATH, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SystemExit("data/records.csv does not have the expected header")
        for row in reader:
            ids.append(row["record_id"])
            features.append([float(row[name]) for name in FEATURE_NAMES])
            outcome.append(int(float(row["churned"])))
            value.append(float(row["order_value_inr_hundreds"]))

    X = np.asarray(features, dtype=float)
    y = np.asarray(outcome, dtype=float)
    v = np.asarray(value, dtype=float)
    return ids, X, y, v


# --------------------------------------------------------------------------------------
# Descriptive statistics (sample form, divisor n - 1)
# --------------------------------------------------------------------------------------
def mean(a: np.ndarray) -> float:
    return float(np.sum(a) / a.shape[0])


def var(a: np.ndarray) -> float:
    """Sample variance, divisor n - 1."""
    n = a.shape[0]
    m = mean(a)
    return float(np.sum((a - m) ** 2) / (n - 1))


def sd(a: np.ndarray) -> float:
    return float(np.sqrt(var(a)))


def covariance(a: np.ndarray, b: np.ndarray) -> float:
    """Sample covariance, divisor n - 1."""
    n = a.shape[0]
    return float(np.sum((a - mean(a)) * (b - mean(b))) / (n - 1))


# --------------------------------------------------------------------------------------
# The pinned stratified 70/30 split and the pinned standardisation
# --------------------------------------------------------------------------------------
def stratified_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The split pinned by Task 2.1.

    Within each class, in file order, the first ``floor(7 * count / 10)`` rows go
    to training and the remainder to test.  Both sets stay in file order.
    """
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in (0, 1):
        members = np.flatnonzero(y == cls)  # file order, ascending
        cut = (TRAIN_FRACTION_NUM * members.shape[0]) // TRAIN_FRACTION_DEN
        train_idx.extend(members[:cut].tolist())
        test_idx.extend(members[cut:].tolist())
    return np.asarray(sorted(train_idx)), np.asarray(sorted(test_idx))


def standardise(train: np.ndarray, test: np.ndarray):
    """Standardise with the *training* mean and training sample SD only."""
    mu = train.mean(axis=0)
    sigma = train.std(axis=0, ddof=1)
    return (train - mu) / sigma, (test - mu) / sigma, mu, sigma


def prepare():
    """Load the data and build every shared artefact the Parts reuse."""
    ids, X, y, value = load_records()
    train_idx, test_idx = stratified_split(y)

    Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
    Ztr, Zte, mu, sigma = standardise(Xtr_raw, Xte_raw)

    return {
        "ids": ids,
        "X": X,
        "y": y,
        "value": value,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "Xtr_raw": Xtr_raw,
        "Xte_raw": Xte_raw,
        "ytr": y[train_idx],
        "yte": y[test_idx],
        "Ztr": Ztr,
        "Zte": Zte,
        "mu": mu,
        "sigma": sigma,
    }


# --------------------------------------------------------------------------------------
# Logistic regression, written from scratch
# --------------------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy; the loss pinned by Task 2.2."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def train_logistic(Z: np.ndarray, y: np.ndarray, eta: float = 0.1, updates: int = 1000):
    """Full-batch gradient descent on the mean cross-entropy.

    Starts from all-zero weights and bias, exactly as Task 2.2 requires.
    Returns the weights, the bias, the loss recorded before any update and a
    dictionary of the loss after update counts that were requested.
    """
    n, d = Z.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    p = sigmoid(Z @ w + b)
    loss0 = binary_cross_entropy(y, p)
    losses: dict[int, float] = {0: loss0}

    for step in range(1, updates + 1):
        p = sigmoid(Z @ w + b)
        residual = p - y
        gw = (Z.T @ residual) / n
        gb = float(np.sum(residual) / n)
        w = w - eta * gw
        b = b - eta * gb

        if step in (1, 10, 100, 200, 1000):
            losses[step] = binary_cross_entropy(y, sigmoid(Z @ w + b))

    return w, b, losses


def predict_proba(Z: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(Z @ w + b)


# --------------------------------------------------------------------------------------
# Confusion matrix and the metrics built from it
# --------------------------------------------------------------------------------------
def confusion(y: np.ndarray, predicted_positive: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` for boolean predictions."""
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    y = np.asarray(y, dtype=bool)
    tp = int(np.sum(predicted_positive & y))
    fp = int(np.sum(predicted_positive & ~y))
    tn = int(np.sum(~predicted_positive & ~y))
    fn = int(np.sum(~predicted_positive & y))
    return tp, fp, tn, fn


def safe_div(num: float, den: float) -> float:
    """Division that returns 0 when the denominator is 0 (pinned convention)."""
    return float(num) / float(den) if den else 0.0


def metrics_from_cells(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and specificity from the four cells."""
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def cost_of(fp: int, fn: int) -> float:
    """Money cost of a set of errors, in INR."""
    return COST_FALSE_ALARM * fp + COST_MISSED_CHURN * fn


# --------------------------------------------------------------------------------------
# k-nearest neighbours, written from scratch
# --------------------------------------------------------------------------------------
def pairwise_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance from every row of ``A`` to every row of ``B``."""
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def knn_neighbour_labels(distances: np.ndarray, y_ref: np.ndarray, k: int) -> np.ndarray:
    """Labels of the k nearest references.

    Ties are broken by position in the reference block, which a stable sort
    preserves, so the ranking is exactly ``(distance, reference position)``.
    """
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return y_ref[order]


def knn_predict(distances: np.ndarray, y_ref: np.ndarray, k: int):
    """Positive when at least ``(k + 1) / 2`` of the k neighbours are positive.

    Returns the boolean predictions and the neighbour share used as the
    probability in Part 4's soft vote.
    """
    labels = knn_neighbour_labels(distances, y_ref, k)
    share = (np.sum(labels, axis=1) / k).astype(float)
    needed = (k + 1) / 2.0
    return np.sum(labels, axis=1) >= needed, share


def knn_cross_validation(Ztr: np.ndarray, ytr: np.ndarray, ks: Sequence[int], folds: int = 5):
    """5-fold cross-validated mean F1 for each candidate K.

    The fold of a training row is its 0-based position in the training set modulo
    five.  Distances and features are computed once from the full training block,
    so no fold ever recomputes a mean or a standard deviation.
    """
    n = Ztr.shape[0]
    fold_of_row = np.arange(n) % folds
    distances = pairwise_euclidean(Ztr, Ztr)

    scores: list[tuple[int, float]] = []
    for k in ks:
        fold_f1 = []
        for fold in range(folds):
            held_out = fold_of_row == fold
            reference = ~held_out
            pred, _ = knn_predict(
                distances[np.ix_(held_out, reference)], ytr[reference], k
            )
            tp, fp, tn, fn = confusion(ytr[held_out], pred)
            fold_f1.append(metrics_from_cells(tp, fp, tn, fn)["f1"])
        scores.append((k, float(np.mean(fold_f1))))
    return scores


def best_k(scores: Sequence[tuple[int, float]]) -> int:
    """Highest mean F1; ties go to the smaller K."""
    best = None
    for k, score in scores:
        if best is None or score > best[1] or (score == best[1] and k < best[0]):
            best = (k, score)
    assert best is not None
    return int(best[0])


def positive_at_threshold(probabilities: np.ndarray, tau: float) -> np.ndarray:
    """Positive when the probability is greater than or equal to tau."""
    return np.asarray(probabilities) >= tau


# --------------------------------------------------------------------------------------
# Small shared reporting helpers
# --------------------------------------------------------------------------------------
def banner(title: str) -> None:
    """A readable console banner so a run of the pipeline is self-explanatory."""
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def summarise_results(path: str) -> None:
    """Print how many keys a results file holds, for a quick sanity check."""
    with open(path) as handle:
        count = sum(1 for line in handle if line.strip())
    print(f"  wrote {os.path.basename(path)} ({count} keys)")
