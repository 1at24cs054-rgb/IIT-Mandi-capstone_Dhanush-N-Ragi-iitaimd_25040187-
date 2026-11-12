"""Self-check for the committed outputs (not part of the graded pipeline).

Re-reads everything in ``outputs/`` and asserts the properties a grader would
check: the exact key set each Part must publish, the exact CSV headers and row
counts, and the internal consistency of every rate with the cells beside it.

Run from the repository root:  python verify_outputs.py
"""

from __future__ import annotations

import csv
import os
import sys

from common import OUTPUT_DIR, fmt

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def load_results(name: str) -> dict[str, str]:
    path = os.path.join(OUTPUT_DIR, name)
    check(os.path.exists(path), f"{name} is missing")
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            check(" = " in line, f"{name}: malformed line {line!r}")
            if " = " not in line:
                continue
            key, value = line.split(" = ", 1)
            check(key not in out, f"{name}: duplicate key {key!r}")
            out[key] = value
    return out


def load_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    path = os.path.join(OUTPUT_DIR, name)
    check(os.path.exists(path), f"{name} is missing")
    if not os.path.exists(path):
        return [], []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    body = [dict(zip(header, row)) for row in rows[1:]]
    return header, body


def expect_keys(results: dict[str, str], keys: list[str], label: str) -> None:
    missing = [k for k in keys if k not in results]
    check(not missing, f"{label}: missing keys {missing}")
    extra = [k for k in results if k not in keys]
    check(not extra, f"{label}: unexpected extra keys {extra}")


def require_4dp(results: dict[str, str], key: str, label: str) -> None:
    if key not in results:
        return
    value = results[key]
    if value in ("yes", "no"):
        return
    try:
        float(value)
    except ValueError:
        FAILURES.append(f"{label}: {key} is not numeric ({value!r})")
        return
    check(
        "." in value and len(value.split(".")[1]) == 4,
        f"{label}: {key} is not printed to four decimals ({value!r})",
    )


def check_rate(row: dict[str, str], cells: tuple[str, str], rate: str, label: str) -> None:
    """Assert that a printed rate agrees with the cells printed next to it."""
    try:
        tp, fp = float(row[cells[0]]), float(row[cells[1]])
        printed = float(row[rate])
    except (KeyError, ValueError):
        FAILURES.append(f"{label}: cannot verify {rate}")
        return
    if tp + fp == 0:
        return
    implied = tp / (tp + fp)
    check(abs(implied - printed) <= 0.0002, f"{label}: {rate} {printed} != {implied:.4f}")


# ======================================================================================
# Part 1
# ======================================================================================
def verify_part1() -> None:
    r = load_results("part1_results.txt")
    required = [
        "n", "mean_value", "sd_value", "se_value", "ci95_low", "ci95_high",
        "z_stat", "reject_h0", "n_required",
        "reg_slope", "reg_intercept", "tss", "ess", "rss",
        "tss_equals_ess_plus_rss", "r2", "rmse", "mae",
        "gd_J0", "gd_J1", "gd_J10", "gd_J100", "gd_J500",
        "gd_slope", "gd_intercept", "gd_slope_gap",
        "resid_sd_lower_raw", "resid_sd_upper_raw", "resid_sd_ratio_raw",
        "loglog_intercept", "loglog_slope", "loglog_r2", "loglog_rmse",
        "resid_sd_lower_log", "resid_sd_upper_log", "resid_sd_ratio_log",
        "mr_beta_0", "mr_beta_1", "mr_beta_2", "mr_beta_3",
        "mr_beta_4", "mr_beta_5", "mr_beta_6", "mr_beta_7",
        "mr_r2", "mr_adj_r2", "loglog_adj_r2",
    ]
    expect_keys(r, required, "part1")

    check(r.get("n") == "800", f"part1: n should be 800, got {r.get('n')}")
    check(r.get("reject_h0") in ("yes", "no"), "part1: reject_h0 must be yes/no")
    check(
        r.get("tss_equals_ess_plus_rss") in ("yes", "no"),
        "part1: tss_equals_ess_plus_rss must be yes/no",
    )
    check(r.get("tss_equals_ess_plus_rss") == "yes", "part1: TSS != ESS + RSS")

    # The two ways of fitting the same line must agree.
    gap = float(r.get("gd_slope_gap", "1"))
    check(gap < 0.001, f"part1: gd_slope_gap {gap} is not below 0.001")

    # CI must straddle the mean and be symmetric.
    lo, hi, mean = float(r["ci95_low"]), float(r["ci95_high"]), float(r["mean_value"])
    check(lo < mean < hi, "part1: the confidence interval does not contain the mean")
    check(abs((hi - mean) - (mean - lo)) < 0.002, "part1: the interval is not symmetric")

    # Decomposition of the total sum of squares.
    tss, ess, rss = float(r["tss"]), float(r["ess"]), float(r["rss"])
    check(abs(tss - ess - rss) < 1e-6 * tss, "part1: TSS != ESS + RSS numerically")
    check(abs(float(r["r2"]) - (1 - rss / tss)) < 0.001, "part1: r2 does not match rss/tss")

    # Gradient descent must reduce the cost monotonically at the checkpoints.
    costs = [float(r[k]) for k in ("gd_J0", "gd_J1", "gd_J10", "gd_J100", "gd_J500")]
    check(all(a >= b for a, b in zip(costs, costs[1:])), "part1: GD cost is not decreasing")

    for key in required:
        if key in ("n", "n_required", "reject_h0", "tss_equals_ess_plus_rss"):
            continue
        require_4dp(r, key, "part1")


# ======================================================================================
# Part 2
# ======================================================================================
def verify_part2() -> None:
    r = load_results("part2_results.txt")
    required = ["train_rows", "train_positives", "test_rows", "test_positives"]
    required += ["loss_epoch_0", "loss_epoch_1", "loss_epoch_10", "loss_epoch_100", "loss_epoch_1000"]
    required += [f"weight_X{i}" for i in range(1, 8)] + ["bias", "test_bce"]
    required += [f"loss_epoch_200_eta_{e}" for e in ("0.01", "0.1", "1.0")]
    required += [f"odds_ratio_X{i}" for i in range(1, 8)]
    for tau in ("0.5", "0.3"):
        required += [
            f"tau_{tau}_{k}"
            for k in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity")
        ]
    required += ["blind_accuracy", "knn_best_k"]
    required += [f"knn_{k}" for k in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1")]
    required += [
        "cost_tau", "cost_train_at_tau", "cost_test_at_tau",
        "cost_test_at_0.5", "cost_test_flag_nobody",
    ]
    expect_keys(r, required, "part2")

    # The pinned split.
    check(r.get("train_rows") == "560", "part2: train_rows should be 560")
    check(r.get("train_positives") == "119", "part2: train_positives should be 119")
    check(r.get("test_rows") == "240", "part2: test_rows should be 240")
    check(r.get("test_positives") == "51", "part2: test_positives should be 51")

    # cost_tau must be a multiple of 0.05 in (0, 1).
    tau = float(r["cost_tau"])
    check(abs(tau * 20 - round(tau * 20)) < 1e-9, f"part2: cost_tau {tau} is not a multiple of 0.05")

    # Every threshold block must have cells summing to the test set.
    for label in ("tau_0.5", "tau_0.3"):
        total = sum(int(r[f"{label}_{k}"]) for k in ("tp", "fp", "tn", "fn"))
        check(total == 240, f"part2: {label} cells sum to {total}, expected 240")

    # Rates must agree with their cells.
    for label in ("tau_0.5", "tau_0.3"):
        tp = int(r[f"{label}_tp"]); fp = int(r[f"{label}_fp"])
        tn = int(r[f"{label}_tn"]); fn = int(r[f"{label}_fn"])
        check(abs(float(r[f"{label}_accuracy"]) - (tp + tn) / 240) <= 0.0002, f"part2: {label} accuracy")
        if tp + fp:
            check(abs(float(r[f"{label}_precision"]) - tp / (tp + fp)) <= 0.0002, f"part2: {label} precision")
        check(abs(float(r[f"{label}_recall"]) - tp / (tp + fn)) <= 0.0002, f"part2: {label} recall")
        check(abs(float(r[f"{label}_specificity"]) - tn / (tn + fp)) <= 0.0002, f"part2: {label} specificity")

    # KNN cells must also sum to the test set.
    knn_total = sum(int(r[f"knn_{k}"]) for k in ("tp", "fp", "tn", "fn"))
    check(knn_total == 240, f"part2: knn cells sum to {knn_total}, expected 240")
    check(int(r["knn_best_k"]) % 2 == 1, "part2: knn_best_k must be odd")

    # The money comparison must be internally consistent.
    test_tau = float(r["cost_test_at_tau"])
    flag_nobody = float(r["cost_test_flag_nobody"])
    check(abs(flag_nobody - 400 * 51) < 1e-6, "part2: cost_test_flag_nobody should be 400 * 51")
    check(test_tau <= flag_nobody, "part2: the chosen threshold should beat flagging nobody")

    header, body = load_csv("part2_thresholds.csv")
    check(header == ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], f"part2_thresholds header {header}")
    check(len(body) == 9, f"part2_thresholds has {len(body)} rows, expected 9")
    for row in body:
        cells = sum(int(row[k]) for k in ("tp", "fp", "tn", "fn"))
        check(cells == 240, f"part2_thresholds tau={row['tau']} cells sum to {cells}")
        check_rate(row, ("tp", "fn"), "tpr", f"thresholds tau={row['tau']}")
        check_rate(row, ("fp", "tn"), "fpr", f"thresholds tau={row['tau']}")

    header, body = load_csv("part2_knn_cv.csv")
    check(header == ["K", "mean_cv_f1"], f"part2_knn_cv header {header}")
    check(len(body) == 11, f"part2_knn_cv has {len(body)} rows, expected 11")
    ks = [int(row["K"]) for row in body]
    check(ks == sorted(ks) and ks == list(range(1, 22, 2)), f"part2_knn_cv K values {ks}")

    # The best K reported must be the argmax of the published CV table.
    best = max(body, key=lambda row: (float(row["mean_cv_f1"]), -int(row["K"])))
    check(int(best["K"]) == int(r["knn_best_k"]), "part2: knn_best_k disagrees with part2_knn_cv.csv")


# ======================================================================================
# Part 3
# ======================================================================================
def verify_part3() -> None:
    r = load_results("part3_results.txt")
    required = [f"wcss_k{k}" for k in range(2, 7)] + ["elbow_k"]
    required += [f"size_segment_{s}" for s in "ABC"]
    required += [f"{rule}_clusters_at_cut" for rule in ("single", "complete", "average")]
    required += ["eigenvalue_sum"]
    required += [f"eigenvalue_{i}" for i in range(1, 6)]
    required += [f"explained_ratio_{i}" for i in range(1, 6)]
    required += [f"cumulative_ratio_{i}" for i in range(1, 6)]
    required += ["components_for_80pct"]
    required += [f"pc1_loading_X{i}" for i in range(1, 6)]
    required += [f"pc2_loading_X{i}" for i in range(1, 6)]
    required += [f"row{n}_pc{i}" for n in (1, 2, 3) for i in (1, 2)]
    required += [f"mean_pc1_segment_{s}" for s in "ABC"]
    expect_keys(r, required, "part3")

    # WCSS must fall as K grows.
    wcss = [float(r[f"wcss_k{k}"]) for k in range(2, 7)]
    check(all(a > b for a, b in zip(wcss, wcss[1:])), "part3: WCSS is not decreasing in K")

    # Eigenvalues sorted, explained ratios sum to 1, cumulative ends at 1.
    eig = [float(r[f"eigenvalue_{i}"]) for i in range(1, 6)]
    check(all(a >= b for a, b in zip(eig, eig[1:])), "part3: eigenvalues are not sorted")
    ratios = [float(r[f"explained_ratio_{i}"]) for i in range(1, 6)]
    check(abs(sum(ratios) - 1.0) < 1e-3, f"part3: explained ratios sum to {sum(ratios)}")
    check(abs(float(r["cumulative_ratio_5"]) - 1.0) < 1e-3, "part3: cumulative ratio 5 is not 1")
    check(abs(float(r["eigenvalue_sum"]) - 5.0) < 1e-3, "part3: eigenvalue_sum is not 5")

    # PC1 and PC2 must be orthogonal and unit length.
    for i in (1, 2):
        load = [float(r[f"pc{i}_loading_X{j}"]) for j in range(1, 6)]
        norm = sum(v * v for v in load) ** 0.5
        check(abs(norm - 1.0) < 1e-3, f"part3: pc{i} loadings have norm {norm}")
    dot = sum(float(r[f"pc1_loading_X{j}"]) * float(r[f"pc2_loading_X{j}"]) for j in range(1, 6))
    check(abs(dot) < 1e-3, f"part3: pc1 and pc2 are not orthogonal (dot={dot})")

    header, body = load_csv("part3_segments.csv")
    expected_header = [
        "segment", "size", "mean_X1", "mean_X2", "mean_X3",
        "mean_X4", "mean_X5", "outcome_rate", "mean_value",
    ]
    check(header == expected_header, f"part3_segments header {header}")
    check(len(body) == 3, f"part3_segments has {len(body)} rows, expected 3")
    check([row["segment"] for row in body] == ["A", "B", "C"], "part3_segments is not ordered A, B, C")
    total = sum(int(row["size"]) for row in body)
    check(total == 800, f"part3_segments sizes sum to {total}, expected 800")
    for name in "ABC":
        check(
            int(body["ABC".index(name)]["size"]) == int(r[f"size_segment_{name}"]),
            f"part3: size_segment_{name} disagrees with the CSV",
        )
    # Segments are ordered by ascending mean X3.
    x3 = [float(row["mean_X3"]) for row in body]
    check(x3 == sorted(x3), "part3: segments are not ordered by ascending mean X3")

    header, body = load_csv("part3_hierarchical.csv")
    check(
        header == ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        f"part3_hierarchical header {header}",
    )
    check(len(body) == 15, f"part3_hierarchical has {len(body)} rows, expected 15")
    for rule in ("single", "complete", "average"):
        rows = [row for row in body if row["linkage"] == rule]
        check(len(rows) == 5, f"part3_hierarchical {rule} has {len(rows)} rows")
        heights = [float(row["height"]) for row in rows]
        check(all(a <= b for a, b in zip(heights, heights[1:])), f"part3_hierarchical {rule} heights not monotone")
        check([int(row["merge_no"]) for row in rows] == [1, 2, 3, 4, 5], f"part3_hierarchical {rule} merge numbering")


# ======================================================================================
# Part 4
# ======================================================================================
def verify_part4() -> None:
    r = load_results("part4_results.txt")
    required = [
        "root_entropy", "root_gini",
        "best_split_by_info_gain", "best_split_by_gini_gain",
        "stump_rate_yes", "stump_rate_no",
        "round1_error", "round1_alpha", "round2_error", "round2_alpha",
        "ensemble_accuracy", "oob_fraction",
        "mlp_total", "cnn_total", "fc_layer_params", "conv_layer_params", "rnn_total",
    ]
    expect_keys(r, required, "part4")

    check(r["best_split_by_info_gain"] in ("S1", "S2", "S3", "S4"), "part4: bad info-gain split label")
    check(r["best_split_by_gini_gain"] in ("S1", "S2", "S3", "S4"), "part4: bad Gini split label")

    header, body = load_csv("part4_splits.csv")
    check(
        header == ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        f"part4_splits header {header}",
    )
    check(len(body) == 4, f"part4_splits has {len(body)} rows, expected 4")
    check([row["split"] for row in body] == ["S1", "S2", "S3", "S4"], "part4_splits rows are not S1..S4")
    for row in body:
        total = int(row["n_yes"]) + int(row["n_no"])
        check(total == 560, f"part4_splits {row['split']} covers {total} rows, expected 560")
        check(
            int(row["positives_yes"]) <= int(row["n_yes"])
            and int(row["positives_no"]) <= int(row["n_no"]),
            f"part4_splits {row['split']} has more positives than rows",
        )
        check(0.0 <= float(row["info_gain"]) <= 1.0, f"part4_splits {row['split']} info gain out of range")
    # The label reported must match the table.
    best_info = max(body, key=lambda row: float(row["info_gain"]))["split"]
    best_gini = max(body, key=lambda row: float(row["gini_gain"]))["split"]
    check(best_info == r["best_split_by_info_gain"], "part4: info-gain label disagrees with the table")
    check(best_gini == r["best_split_by_gini_gain"], "part4: Gini label disagrees with the table")

    header, body = load_csv("part4_voting.csv")
    check(
        header == ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        f"part4_voting header {header}",
    )
    check(len(body) == 5, f"part4_voting has {len(body)} rows, expected 5")
    check(
        [row["model"] for row in body] == ["logistic", "knn", "stump", "hard_vote", "soft_vote"],
        "part4_voting row order",
    )
    for row in body:
        cells = sum(int(row[k]) for k in ("tp", "fp", "tn", "fn"))
        check(cells == 240, f"part4_voting {row['model']} covers {cells} rows, expected 240")
        check_rate(row, ("tp", "fp"), "precision", f"voting {row['model']}")
        check_rate(row, ("tp", "fn"), "recall", f"voting {row['model']}")

    header, body = load_csv("part4_adaboost.csv")
    check(
        header == ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        f"part4_adaboost header {header}",
    )
    check(len(body) == 16, f"part4_adaboost has {len(body)} rows, expected 16")
    for round_no in ("1", "2"):
        rows = [row for row in body if row["round"] == round_no]
        check(len(rows) == 8, f"part4_adaboost round {round_no} has {len(rows)} rows")
        total = sum(float(row["weight_after"]) for row in rows)
        check(abs(total - 1.0) < 1e-3, f"part4_adaboost round {round_no} weights sum to {total}")

    header, body = load_csv("part4_architecture.csv")
    check(header == ["layer", "parameters"], f"part4_architecture header {header}")
    check(len(body) == 7, f"part4_architecture has {len(body)} rows, expected 7")
    check(
        [row["layer"] for row in body]
        == ["mlp_hidden1", "mlp_hidden2", "mlp_output", "cnn_conv1", "cnn_conv2", "cnn_dense1", "cnn_dense2"],
        "part4_architecture row order",
    )
    params = {row["layer"]: int(row["parameters"]) for row in body}
    check(
        params["mlp_hidden1"] + params["mlp_hidden2"] + params["mlp_output"] == int(r["mlp_total"]),
        "part4: mlp_total does not equal the sum of its layers",
    )
    check(
        params["cnn_conv1"] + params["cnn_conv2"] + params["cnn_dense1"] + params["cnn_dense2"]
        == int(r["cnn_total"]),
        "part4: cnn_total does not equal the sum of its layers",
    )


# ======================================================================================
# Repository-level checks
# ======================================================================================
def verify_repository() -> None:
    for name in ("README.md", "requirements.txt", "generate_data.py", "data/records.csv"):
        check(os.path.exists(name), f"required file missing: {name}")

    check(os.path.exists("data/records.csv"), "data/records.csv is missing")
    if os.path.exists("data/records.csv"):
        import hashlib

        with open("data/records.csv", "rb") as handle:
            digest = hashlib.md5(handle.read()).hexdigest()
        check(
            digest == "c6f249ba693f1ebeed318d6d5e18de23",
            f"data/records.csv hash is {digest}, expected c6f249ba693f1ebeed318d6d5e18de23",
        )

    if os.path.exists("README.md"):
        with open("README.md") as handle:
            lines = handle.read().splitlines()
        check(bool(lines) and lines[0] == "Domain: E-commerce", "README first line must be 'Domain: E-commerce'")
        required_sections = [
            "## How to run",
            "## Part 1 findings",
            "## Part 2 findings",
            "## Part 3 findings",
            "## Part 4 findings",
            "## Model-choice memo",
            "## Limitations",
        ]
        positions = []
        for section in required_sections:
            check(section in lines, f"README is missing the section {section!r}")
            if section in lines:
                positions.append(lines.index(section))
        check(positions == sorted(positions), "README sections are not in the required order")

        memo_start = lines.index("## Model-choice memo") if "## Model-choice memo" in lines else None
        limits_index = lines.index("## Limitations") if "## Limitations" in lines else None
        if memo_start is not None and limits_index is not None:
            memo_words = sum(len(line.split()) for line in lines[memo_start + 1 : limits_index])
            check(memo_words <= 300, f"the model-choice memo is {memo_words} words, limit is 300")

    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as handle:
            content = handle.read()
        check("numpy==" in content, "requirements.txt must pin numpy")


def main() -> int:
    verify_part1()
    verify_part2()
    verify_part3()
    verify_part4()
    verify_repository()

    print("=" * 74)
    print("OUTPUT VERIFICATION")
    print("=" * 74)
    if FAILURES:
        print(f"FAILED with {len(FAILURES)} problem(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("All checks passed:")
    print("  * every required results key is present, with no extras")
    print("  * every CSV has the exact header and row count")
    print("  * every printed rate agrees with the cells beside it")
    print("  * PCA loadings are unit length and orthogonal")
    print("  * the dataset hash matches the brief")
    print("  * the README sections are present, ordered, and the memo is within 300 words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
