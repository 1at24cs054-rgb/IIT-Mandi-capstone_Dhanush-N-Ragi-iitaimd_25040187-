"""Part 2 - Classification and evaluation (30 marks).

Builds a churn-probability model for X1..X7 and judges it honestly:

* logistic regression from scratch (full-batch gradient descent),
* a K sweep for k-nearest neighbours chosen by 5-fold cross-validation,
* a full threshold sweep reported as ROC points,
* an alert threshold chosen on the training rows with a money cost.

Run from the repository root:  python part2.py
Writes:  outputs/part2_results.txt
         outputs/part2_thresholds.csv
         outputs/part2_knn_cv.csv
"""

from __future__ import annotations

import numpy as np

from common import (
    COST_FALSE_ALARM,
    COST_MISSED_CHURN,
    Results,
    banner,
    best_k as pick_best_k,
    binary_cross_entropy,
    confusion,
    cost_of,
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

LEARNING_RATE = 0.1
UPDATES = 1000
ETA_SWEEP = (0.01, 0.1, 1.0)
ETA_SWEEP_UPDATES = 200
TAU_MAIN = 0.5
TAU_ALT = 0.3
SWEEP_TAUS = tuple(round(0.1 * k, 1) for k in range(1, 10))
K_CANDIDATES = tuple(range(1, 22, 2))
FOLDS = 5


def evaluate(y: np.ndarray, probabilities: np.ndarray, tau: float) -> dict[str, float]:
    """Confusion cells and the five metrics at one threshold."""
    predicted = positive_at_threshold(probabilities, tau)
    tp, fp, tn, fn = confusion(y, predicted)
    out = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    out.update(metrics_from_cells(tp, fp, tn, fn))
    return out


def main() -> None:
    data = prepare()
    ytr, yte = data["ytr"], data["yte"]
    Ztr, Zte = data["Ztr"], data["Zte"]

    banner("PART 2 - Classification and evaluation")

    results = Results()

    # ---------------------------------------------------------------- 2.1 Split
    results.add_many(
        train_rows=int(ytr.shape[0]),
        train_positives=int(np.sum(ytr)),
        test_rows=int(yte.shape[0]),
        test_positives=int(np.sum(yte)),
    )
    print(
        f"  split: {int(ytr.shape[0])} train ({int(np.sum(ytr))} positives), "
        f"{int(yte.shape[0])} test ({int(np.sum(yte))} positives)"
    )

    # ---------------------------------------------------------------- 2.2 Logistic regression
    w, b, losses = train_logistic(Ztr, ytr, eta=LEARNING_RATE, updates=UPDATES)

    train_prob = predict_proba(Ztr, w, b)
    test_prob = predict_proba(Zte, w, b)

    test_bce = binary_cross_entropy(yte, test_prob)

    results.add_many(
        loss_epoch_0=losses[0],
        loss_epoch_1=losses[1],
        loss_epoch_10=losses[10],
        loss_epoch_100=losses[100],
        loss_epoch_1000=losses[1000],
    )
    for index in range(7):
        results.add(f"weight_X{index + 1}", float(w[index]))
    results.add_many(bias=float(b), test_bce=test_bce)

    print(f"  logistic: loss {losses[0]:.4f} -> {losses[1000]:.4f}, test BCE {test_bce:.4f}")

    # ---------------------------------------------------------------- 2.3 Learning rate
    eta_losses = {}
    for eta in ETA_SWEEP:
        _, _, path = train_logistic(Ztr, ytr, eta=eta, updates=ETA_SWEEP_UPDATES)
        eta_losses[eta] = path[ETA_SWEEP_UPDATES]
        results.add(f"loss_epoch_200_eta_{eta}", eta_losses[eta])
    ordered = sorted(eta_losses.items(), key=lambda kv: kv[1])
    print("  eta sweep (lowest first): " + ", ".join(f"{e}:{v:.4f}" for e, v in ordered))

    # ---------------------------------------------------------------- 2.4 Odds ratios
    odds = np.exp(w)
    for index in range(7):
        results.add(f"odds_ratio_X{index + 1}", float(odds[index]))
    print("  odds ratios: " + ", ".join(f"X{i + 1}={v:.3f}" for i, v in enumerate(odds)))

    # ---------------------------------------------------------------- 2.5 Two thresholds
    main_eval = evaluate(yte, test_prob, TAU_MAIN)
    for key, value in main_eval.items():
        results.add(f"tau_0.5_{key}", value)

    alt_eval = evaluate(yte, test_prob, TAU_ALT)
    for key, value in alt_eval.items():
        results.add(f"tau_0.3_{key}", value)

    blind_tp, blind_fp, blind_tn, blind_fn = confusion(yte, np.zeros_like(yte, dtype=bool))
    blind_accuracy = metrics_from_cells(blind_tp, blind_fp, blind_tn, blind_fn)["accuracy"]
    results.add("blind_accuracy", blind_accuracy)

    print(
        f"  tau 0.5: acc {main_eval['accuracy']:.4f} prec {main_eval['precision']:.4f} "
        f"rec {main_eval['recall']:.4f} f1 {main_eval['f1']:.4f}"
    )
    print(
        f"  tau 0.3: acc {alt_eval['accuracy']:.4f} prec {alt_eval['precision']:.4f} "
        f"rec {alt_eval['recall']:.4f} f1 {alt_eval['f1']:.4f}"
    )

    # ---------------------------------------------------------------- 2.6 Threshold sweep
    rows = []
    for tau in SWEEP_TAUS:
        cell = evaluate(yte, test_prob, tau)
        tpr = cell["tp"] / (cell["tp"] + cell["fn"]) if (cell["tp"] + cell["fn"]) else 0.0
        fpr = cell["fp"] / (cell["fp"] + cell["tn"]) if (cell["fp"] + cell["tn"]) else 0.0
        rows.append([tau, cell["tp"], cell["fp"], cell["tn"], cell["fn"], tpr, fpr])
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)
    print(f"  wrote part2_thresholds.csv ({len(rows)} ROC points)")

    # ---------------------------------------------------------------- 2.7 KNN + CV
    cv_scores = knn_cross_validation(Ztr, ytr, K_CANDIDATES, folds=FOLDS)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[k, s] for k, s in cv_scores])

    chosen_k = pick_best_k(cv_scores)
    results.add("knn_best_k", chosen_k)

    test_distances = pairwise_euclidean(Zte, Ztr)
    knn_pred, knn_share = knn_predict(test_distances, ytr, chosen_k)
    ktp, kfp, ktn, kfn = confusion(yte, knn_pred)
    knn_metrics = metrics_from_cells(ktp, kfp, ktn, kfn)

    results.add_many(knn_tp=ktp, knn_fp=kfp, knn_tn=ktn, knn_fn=kfn)
    for key in ("accuracy", "precision", "recall", "f1"):
        results.add(f"knn_{key}", knn_metrics[key])

    print(
        f"  knn: best K={chosen_k}, acc {knn_metrics['accuracy']:.4f} "
        f"prec {knn_metrics['precision']:.4f} rec {knn_metrics['recall']:.4f} "
        f"f1 {knn_metrics['f1']:.4f}"
    )

    # ---------------------------------------------------------------- 2.8 Cost threshold
    best_tau = None
    best_train_cost = None
    for step in range(1, 20):
        tau = step / 20.0
        tp, fp, tn, fn = confusion(ytr, positive_at_threshold(train_prob, tau))
        cost = cost_of(fp, fn)
        if best_train_cost is None or cost < best_train_cost:
            best_tau, best_train_cost = tau, cost

    test_at_tau = cost_of(*_errors(yte, test_prob, best_tau))
    test_at_half = cost_of(*_errors(yte, test_prob, TAU_MAIN))
    test_flag_nobody = cost_of(0, int(np.sum(yte)))

    # These keys contain dots, so they are added one at a time rather than as
    # keyword arguments.
    results.add("cost_tau", best_tau)
    results.add("cost_train_at_tau", best_train_cost)
    results.add("cost_test_at_tau", test_at_tau)
    results.add("cost_test_at_0.5", test_at_half)
    results.add("cost_test_flag_nobody", test_flag_nobody)

    print(
        f"  money threshold: tau={best_tau:.2f} train cost INR {best_train_cost:,.0f} | "
        f"test cost INR {test_at_tau:,.0f} vs INR {test_at_half:,.0f} at 0.5 "
        f"vs INR {test_flag_nobody:,.0f} doing nothing"
    )

    path = results.write("part2_results.txt")
    summarise_results(path)

    # ---------------------------------------------------------------- console extras
    print(f"  blind accuracy {blind_accuracy:.4f}")
    print(
        "  cost rule: false alarm INR {:,.0f}, missed churner INR {:,.0f}".format(
            COST_FALSE_ALARM, COST_MISSED_CHURN
        )
    )


def _errors(y: np.ndarray, probabilities: np.ndarray, tau: float) -> tuple[int, int]:
    """Return ``(fp, fn)`` for a threshold, the two cells the cost rule needs."""
    tp, fp, tn, fn = confusion(y, positive_at_threshold(probabilities, tau))
    return fp, fn


if __name__ == "__main__":
    main()
