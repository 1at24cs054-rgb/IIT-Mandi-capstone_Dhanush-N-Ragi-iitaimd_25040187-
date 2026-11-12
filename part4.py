"""Part 4 - Model choice and architecture sizing (20 marks).

Works on the Task 2.1 training/test split and reuses the Part 2 models through
``common.py`` (which never re-runs a Part).

* entropy / Gini split analysis and a decision stump,
* hard and soft voting on the test rows,
* two rounds of AdaBoost arithmetic on the supplied toy problem,
* the out-of-bag share,
* parameter counts for the MLP, CNN and RNN the later projects would need.

Run from the repository root:  python part4.py
Writes:  outputs/part4_results.txt
         outputs/part4_splits.csv
         outputs/part4_voting.csv
         outputs/part4_adaboost.csv
         outputs/part4_architecture.csv
"""

from __future__ import annotations

import numpy as np

from common import (
    Results,
    banner,
    best_k as pick_best_k,
    confusion,
    knn_cross_validation,
    knn_predict,
    metrics_from_cells,
    pairwise_euclidean,
    positive_at_threshold,
    prepare,
    predict_proba,
    summarise_results,
    train_logistic,
    write_csv,
)

TAU = 0.3
K_CANDIDATES = tuple(range(1, 22, 2))
FOLDS = 5
OOB_TRAIN_SIZE = 560

# The Task 4.3 toy problem.
Y_TOY = np.array([+1, +1, +1, -1, -1, -1, +1, -1], dtype=float)
H1 = np.array([+1, +1, -1, -1, -1, -1, -1, -1], dtype=float)
H2 = np.array([-1, +1, +1, -1, +1, -1, +1, -1], dtype=float)


# ======================================================================================
# Impurity measures
# ======================================================================================
def entropy(positives: int, total: int) -> float:
    """Binary entropy in bits; 0 for a pure (or empty) node."""
    if total == 0 or positives in (0, total):
        return 0.0
    p = positives / total
    q = 1.0 - p
    return float(-(p * np.log2(p) + q * np.log2(q)))


def gini(positives: int, total: int) -> float:
    """Binary Gini impurity; 0 for a pure (or empty) node."""
    if total == 0:
        return 0.0
    p = positives / total
    q = 1.0 - p
    return float(1.0 - (p * p + q * q))


def split_gains(y: np.ndarray, mask_yes: np.ndarray) -> dict[str, float]:
    """Weighted child impurity and both gains for one yes/no split."""
    total = y.shape[0]
    n_yes = int(np.sum(mask_yes))
    n_no = total - n_yes
    pos_yes = int(np.sum(y[mask_yes]))
    pos_no = int(np.sum(y[~mask_yes]))

    parent_entropy = entropy(int(np.sum(y)), total)
    parent_gini = gini(int(np.sum(y)), total)

    child_entropy = (n_yes / total) * entropy(pos_yes, n_yes) + (
        n_no / total
    ) * entropy(pos_no, n_no)
    child_gini = (n_yes / total) * gini(pos_yes, n_yes) + (n_no / total) * gini(
        pos_no, n_no
    )

    return {
        "n_yes": n_yes,
        "positives_yes": pos_yes,
        "n_no": n_no,
        "positives_no": pos_no,
        "info_gain": parent_entropy - child_entropy,
        "gini_gain": parent_gini - child_gini,
    }


# ======================================================================================
# AdaBoost
# ======================================================================================
def adaboost_two_rounds():
    """The two pinned boosting rounds, with the full weight bookkeeping."""
    n = Y_TOY.shape[0]
    weights = np.full(n, 1.0 / n)
    rows = []
    history = []

    for round_no, hypothesis in ((1, H1), (2, H2)):
        wrong = hypothesis != Y_TOY
        error = float(np.sum(weights[wrong]))
        alpha = 0.5 * np.log((1.0 - error) / error)

        # Keep the pre-update weights for the CSV before overwriting them.
        before = weights.copy()
        unnormalised = before * np.exp(-alpha * Y_TOY * hypothesis)
        after = unnormalised / np.sum(unnormalised)
        weights = after

        history.append((error, alpha))
        for point in range(n):
            rows.append(
                [round_no, point + 1, before[point], unnormalised[point], after[point]]
            )

    combined = history[0][1] * H1 + history[1][1] * H2
    ensemble_predictions = np.sign(combined)
    ensemble_accuracy = float(np.mean(ensemble_predictions == Y_TOY))
    return history, rows, ensemble_accuracy


# ======================================================================================
# Architecture arithmetic
# ======================================================================================
def dense_params(fan_in: int, neurons: int) -> int:
    """A fully connected layer: one weight per input-neuron pair plus one bias."""
    return fan_in * neurons + neurons


def conv_params(filter_height: int, filter_width: int, in_depth: int, filters: int) -> int:
    """A convolution: one weight per filter position per input channel, plus bias."""
    return (filter_height * filter_width * in_depth + 1) * filters


def conv_output_side(side: int, kernel: int, padding: int, stride: int) -> int:
    """Per-side output size of a convolution or pooling layer."""
    return (side - kernel + 2 * padding) // stride + 1


def architecture_rows():
    """Parameter counts for Model A (both options) and the scaling comparison."""
    # Option 1: 784 -> 16 -> 16 -> 10 fully connected.
    mlp_hidden1 = dense_params(784, 16)
    mlp_hidden2 = dense_params(16, 16)
    mlp_output = dense_params(16, 10)

    # Option 2: conv(5x5, 6, pad 2) -> pool -> conv(5x5, 16, pad 0) -> pool -> dense 32 -> 10
    cnn_conv1 = conv_params(5, 5, 1, 6)
    cnn_conv2 = conv_params(5, 5, 6, 16)

    after_pool1 = conv_output_side(conv_output_side(28, 5, 2, 1), 2, 0, 2)  # 28 -> 14
    after_conv2 = conv_output_side(after_pool1, 5, 0, 1)  # 14 -> 10
    after_pool2 = conv_output_side(after_conv2, 2, 0, 2)  # 10 -> 5
    flattened = after_pool2 * after_pool2 * 16

    cnn_dense1 = dense_params(flattened, 32)
    cnn_dense2 = dense_params(32, 10)

    rows = [
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2],
    ]

    shapes = [
        ("conv1", 28, 28, 6, cnn_conv1),
        ("pool1", 14, 14, 6, 0),
        ("conv2", 10, 10, 16, cnn_conv2),
        ("pool2", 5, 5, 16, 0),
        ("flatten", 1, 1, flattened, 0),
        ("dense1", 1, 1, 32, cnn_dense1),
        ("dense2", 1, 1, 10, cnn_dense2),
    ]

    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2
    return rows, mlp_total, cnn_total, shapes, flattened


def main() -> None:
    data = prepare()
    ytr, yte = data["ytr"], data["yte"]
    Ztr, Zte = data["Ztr"], data["Zte"]
    Xtr_raw, Xte_raw = data["Xtr_raw"], data["Xte_raw"]

    banner("PART 4 - Model choice and architecture sizing")

    results = Results()

    # ---------------------------------------------------------------- 4.1 Splits
    total = ytr.shape[0]
    root_entropy = entropy(int(np.sum(ytr)), total)
    root_gini = gini(int(np.sum(ytr)), total)
    results.add_many(root_entropy=root_entropy, root_gini=root_gini)

    candidates = {
        "S1": Xtr_raw[:, 5] == 1,  # X6 cod flag
        "S2": Xtr_raw[:, 6] == 1,  # X7 plus member
        "S3": Xtr_raw[:, 2] > 0.5,  # X3 discount share
        "S4": Xtr_raw[:, 0] <= 24,  # X1 tenure
    }

    split_rows = []
    gains = {}
    for name, mask in candidates.items():
        detail = split_gains(ytr, mask)
        gains[name] = detail
        split_rows.append(
            [
                name,
                detail["n_yes"],
                detail["positives_yes"],
                detail["n_no"],
                detail["positives_no"],
                detail["info_gain"],
                detail["gini_gain"],
            ]
        )

    write_csv(
        "part4_splits.csv",
        ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        split_rows,
    )

    best_info = max(gains, key=lambda name: gains[name]["info_gain"])
    best_gini = max(gains, key=lambda name: gains[name]["gini_gain"])
    results.add_many(best_split_by_info_gain=best_info, best_split_by_gini_gain=best_gini)

    print(f"  root entropy {root_entropy:.4f}  root Gini {root_gini:.4f}")
    print(
        "  info gain: "
        + ", ".join(f"{n}={gains[n]['info_gain']:.4f}" for n in gains)
    )
    print(f"  best split by information gain: {best_info}; by Gini gain: {best_gini}")

    # ---------------------------------------------------------------- 4.2 Stump + votes
    stump_mask_tr = candidates[best_info]
    stump_rate_yes = float(np.mean(ytr[stump_mask_tr]))
    stump_rate_no = float(np.mean(ytr[~stump_mask_tr]))
    results.add_many(stump_rate_yes=stump_rate_yes, stump_rate_no=stump_rate_no)

    stump_mask_te = {
        "S1": Xte_raw[:, 5] == 1,
        "S2": Xte_raw[:, 6] == 1,
        "S3": Xte_raw[:, 2] > 0.5,
        "S4": Xte_raw[:, 0] <= 24,
    }[best_info]
    stump_prob_te = np.where(stump_mask_te, stump_rate_yes, stump_rate_no)
    stump_pred = positive_at_threshold(stump_prob_te, TAU)

    # Rebuild the Part 2 models through the shared helpers.
    w, b, _ = train_logistic(Ztr, ytr, eta=0.1, updates=1000)
    logistic_prob = predict_proba(Zte, w, b)
    logistic_pred = positive_at_threshold(logistic_prob, TAU)

    cv_scores = knn_cross_validation(Ztr, ytr, K_CANDIDATES, folds=FOLDS)
    chosen_k = pick_best_k(cv_scores)
    knn_pred, knn_share = knn_predict(pairwise_euclidean(Zte, Ztr), ytr, chosen_k)

    hard_vote = (
        logistic_pred.astype(int) + knn_pred.astype(int) + stump_pred.astype(int)
    ) >= 2
    soft_vote = positive_at_threshold((logistic_prob + knn_share + stump_prob_te) / 3.0, TAU)

    voting_rows = []
    for name, predicted in (
        ("logistic", logistic_pred),
        ("knn", knn_pred),
        ("stump", stump_pred),
        ("hard_vote", hard_vote),
        ("soft_vote", soft_vote),
    ):
        tp, fp, tn, fn = confusion(yte, predicted)
        m = metrics_from_cells(tp, fp, tn, fn)
        voting_rows.append([name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])

    write_csv(
        "part4_voting.csv",
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )
    print(f"  stump: yes-side churn {stump_rate_yes:.4f}, no-side {stump_rate_no:.4f}")
    print(
        "  voting f1: "
        + ", ".join(f"{row[0]}={row[7]:.4f}" for row in voting_rows)
    )

    # ---------------------------------------------------------------- 4.3 AdaBoost
    history, adaboost_rows, ensemble_accuracy = adaboost_two_rounds()
    results.add_many(
        round1_error=history[0][0],
        round1_alpha=history[0][1],
        round2_error=history[1][0],
        round2_alpha=history[1][1],
        ensemble_accuracy=ensemble_accuracy,
    )
    write_csv(
        "part4_adaboost.csv",
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        adaboost_rows,
    )
    misclassified_weight = float(sum(adaboost_rows[i][4] for i in (2, 6)))
    print(
        f"  adaboost: error {history[0][0]:.4f} -> {history[1][0]:.4f}, "
        f"alpha {history[0][1]:.4f} -> {history[1][1]:.4f}, "
        f"ensemble accuracy {ensemble_accuracy:.4f}"
    )
    print(f"  weight on the two misclassified points after round 1: {misclassified_weight:.4f}")

    # ---------------------------------------------------------------- 4.4 Out-of-bag
    oob_fraction = float((1.0 - 1.0 / OOB_TRAIN_SIZE) ** OOB_TRAIN_SIZE)
    results.add("oob_fraction", oob_fraction)
    print(f"  out-of-bag share: {oob_fraction:.4f}")

    # ---------------------------------------------------------------- 4.5 Architecture
    arch_rows, mlp_total, cnn_total, shapes, flattened = architecture_rows()
    write_csv("part4_architecture.csv", ["layer", "parameters"], arch_rows)
    results.add_many(mlp_total=mlp_total, cnn_total=cnn_total)

    fc_layer_params = dense_params(128 * 128 * 3, 1024)
    conv_layer_params = conv_params(3, 3, 3, 64)
    results.add_many(fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params)

    # Model B: 64-dim input -> 128 hidden -> 16 classes.
    W_XH = 128 * 64
    W_HH = 128 * 128
    B_H = 128
    W_HY = 16 * 128
    B_Y = 16
    results.add("rnn_total", W_XH + W_HH + B_H + W_HY + B_Y)

    path = results.write("part4_results.txt")
    summarise_results(path)

    print(f"  MLP total {mlp_total:,} params; CNN total {cnn_total:,} params")
    for name, height, width, depth, params in shapes:
        print(f"    {name:<8} {height}x{width}x{depth:<4} {params:>7,} params")
    print(f"  dense 128x128x3 -> 1024: {fc_layer_params:,}; conv 64x(3x3x3): {conv_layer_params:,}")
    print(f"  RNN total {W_XH + W_HH + B_H + W_HY + B_Y:,} params")


if __name__ == "__main__":
    main()
