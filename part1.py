"""Part 1 - Estimation, inference and a regression baseline (25 marks).

All 800 rows are used.  Every number below is computed by the code in this file
(or by the shared helpers in ``common.py``); the closed forms are written out
explicitly rather than delegated to a ready-made regression routine.

Run from the repository root:  python part1.py
Writes:                        outputs/part1_results.txt
"""

from __future__ import annotations

import numpy as np

from common import (
    banner,
    covariance,
    load_records,
    mean,
    sd,
    var,
    Results,
    summarise_results,
)

# The planning claim being tested (Task 1.2), in hundreds of INR.
H0_MEAN = 8.0
Z_CRITICAL = 1.96
MARGIN_OF_ERROR = 0.5
LEARNING_RATE = 0.1
GD_UPDATES = 500


def closed_form_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Simple linear regression fitted with the closed-form formulas.

    slope = cov(x, y) / var(x) with both moments in sample form, and the
    intercept follows from the line passing through the means.
    """
    n = x.shape[0]
    slope = covariance(x, y) / var(x)
    intercept = mean(y) - slope * mean(x)
    fitted = intercept + slope * x

    tss = float(np.sum((y - mean(y)) ** 2))
    ess = float(np.sum((fitted - mean(y)) ** 2))
    rss = float(np.sum((y - fitted) ** 2))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "fitted": fitted,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "r2": 1.0 - rss / tss,
        "rmse": float(np.sqrt(rss / n)),
        "mae": float(np.mean(np.abs(y - fitted))),
    }


def gradient_descent(x: np.ndarray, y: np.ndarray, eta: float, updates: int):
    """Full-batch gradient descent on J = (1/n) * sum of squared errors.

    The predictor is standardised first, so the fitted line is ``m * z + c``.
    Returns the parameter path at the checkpoints plus the recorded costs.
    """
    n = x.shape[0]
    z = (x - mean(x)) / sd(x)

    m = 0.0
    c = 0.0

    def cost(m_value: float, c_value: float) -> float:
        residual = y - (m_value * z + c_value)
        return float(np.sum(residual * residual) / n)

    costs = {0: cost(m, c)}
    for step in range(1, updates + 1):
        residual = y - (m * z + c)
        grad_m = -(2.0 / n) * float(np.sum(z * residual))
        grad_c = -(2.0 / n) * float(np.sum(residual))
        m = m - eta * grad_m
        c = c - eta * grad_c
        if step in (1, 10, 100, updates):
            costs[step] = cost(m, c)

    return m, c, z, costs


def half_split_residual_sd(fitted: np.ndarray, residual: np.ndarray):
    """Split rows by ascending fitted value (stable sort) into two halves.

    Ties keep file order because the sort is stable, which is what keeps the
    ratio inside its pass band.
    """
    order = np.argsort(fitted, kind="stable")
    lower = order[: fitted.shape[0] // 2]
    upper = order[fitted.shape[0] // 2 :]
    lower_sd = sd(residual[lower])
    upper_sd = sd(residual[upper])
    return lower_sd, upper_sd, upper_sd / lower_sd


def multiple_regression(design: np.ndarray, target: np.ndarray) -> dict:
    """Least squares through the normal equation, using an explicit solve.

    ``design`` already carries its own constant column, so the coefficients come
    back in the order the caller built the columns.
    """
    n, k = design.shape
    gram = design.T @ design
    rhs = design.T @ target
    beta = np.linalg.solve(gram, rhs)

    fitted = design @ beta
    residual = target - fitted
    tss = float(np.sum((target - mean(target)) ** 2))
    rss = float(np.sum(residual * residual))
    r2 = 1.0 - rss / tss
    p = k - 1  # predictors excluding the constant
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

    return {
        "beta": beta,
        "r2": r2,
        "adj_r2": adj_r2,
        "fitted": fitted,
        "residual": residual,
        "rss": rss,
        "tss": tss,
    }


def main() -> None:
    ids, X, y, value = load_records()
    n = value.shape[0]

    banner("PART 1 - Estimation, inference and a regression baseline")

    results = Results()

    # ---------------------------------------------------------------- 1.1 Interval
    m = mean(value)
    s = sd(value)
    se = s / np.sqrt(n)
    ci_low = m - Z_CRITICAL * se
    ci_high = m + Z_CRITICAL * se

    results.add_many(
        n=n,
        mean_value=m,
        sd_value=s,
        se_value=se,
        ci95_low=ci_low,
        ci95_high=ci_high,
    )

    # ---------------------------------------------------------------- 1.2 Hypothesis test
    z_stat = (m - H0_MEAN) / se
    reject_h0 = abs(z_stat) > Z_CRITICAL
    results.add_many(z_stat=z_stat, reject_h0=reject_h0)

    # ---------------------------------------------------------------- 1.3 Sample size
    n_required = int(np.ceil((Z_CRITICAL * s / MARGIN_OF_ERROR) ** 2))
    results.add("n_required", n_required)

    # ---------------------------------------------------------------- 1.4 Closed form
    x4 = X[:, 3]
    reg = closed_form_regression(x4, value)
    results.add_many(
        reg_slope=reg["slope"],
        reg_intercept=reg["intercept"],
        tss=reg["tss"],
        ess=reg["ess"],
        rss=reg["rss"],
        tss_equals_ess_plus_rss=abs(reg["tss"] - reg["ess"] - reg["rss"])
        < 1e-6 * reg["tss"],
        r2=reg["r2"],
        rmse=reg["rmse"],
        mae=reg["mae"],
    )

    # ---------------------------------------------------------------- 1.5 Gradient descent
    gd_m, gd_c, z4, gd_costs = gradient_descent(
        x4, value, eta=LEARNING_RATE, updates=GD_UPDATES
    )
    sd_x4 = sd(x4)
    mean_x4 = mean(x4)

    gd_slope = gd_m / sd_x4
    gd_intercept = gd_c - gd_m * mean_x4 / sd_x4
    gd_slope_gap = abs(gd_slope - reg["slope"])

    results.add_many(
        gd_J0=gd_costs[0],
        gd_J1=gd_costs[1],
        gd_J10=gd_costs[10],
        gd_J100=gd_costs[100],
        gd_J500=gd_costs[GD_UPDATES],
        gd_slope=gd_slope,
        gd_intercept=gd_intercept,
        gd_slope_gap=gd_slope_gap,
    )

    # ---------------------------------------------------------------- 1.6 Funnel check
    raw_residual = value - reg["fitted"]
    raw_low, raw_high, raw_ratio = half_split_residual_sd(reg["fitted"], raw_residual)

    log_value = np.log(value)
    log_x4 = np.log(x4)
    loglog = closed_form_regression(log_x4, log_value)
    log_residual = log_value - loglog["fitted"]
    log_low, log_high, log_ratio = half_split_residual_sd(
        loglog["fitted"], log_residual
    )
    loglog_adj_r2 = 1.0 - (1.0 - loglog["r2"]) * (n - 1) / (n - 1 - 1)

    results.add_many(
        resid_sd_lower_raw=raw_low,
        resid_sd_upper_raw=raw_high,
        resid_sd_ratio_raw=raw_ratio,
        loglog_intercept=loglog["intercept"],
        loglog_slope=loglog["slope"],
        loglog_r2=loglog["r2"],
        loglog_rmse=loglog["rmse"],
        resid_sd_lower_log=log_low,
        resid_sd_upper_log=log_high,
        resid_sd_ratio_log=log_ratio,
        loglog_adj_r2=loglog_adj_r2,
    )

    # ---------------------------------------------------------------- 1.7 Multiple regression
    design = np.column_stack(
        [
            np.ones(n),  # constant
            log_x4,  # ln(X4)
            X[:, 0],  # X1 tenure
            X[:, 1],  # X2 complaints
            X[:, 2],  # X3 discount share
            X[:, 4],  # X5 loyalty
            X[:, 5],  # X6 cod flag
            X[:, 6],  # X7 plus-member flag
        ]
    )
    multi = multiple_regression(design, log_value)
    for index, coefficient in enumerate(multi["beta"]):
        results.add(f"mr_beta_{index}", float(coefficient))
    results.add_many(mr_r2=multi["r2"], mr_adj_r2=multi["adj_r2"])

    path = results.write("part1_results.txt")
    summarise_results(path)

    # ---------------------------------------------------------------- console report
    print(f"  value mean {m:.4f}  sd {s:.4f}  se {se:.4f}")
    print(f"  95% CI [{ci_low:.4f}, {ci_high:.4f}]  z {z_stat:.4f} -> reject H0: {reject_h0}")
    print(f"  sample size needed for +/-{MARGIN_OF_ERROR}: {n_required} rows")
    print(f"  closed form: slope {reg['slope']:.4f}  intercept {reg['intercept']:.4f}  R2 {reg['r2']:.4f}")
    print(f"  gradient descent after {GD_UPDATES} updates: gap {gd_slope_gap:.8f}")
    print(f"  residual SD ratio  raw {raw_ratio:.4f}  log {log_ratio:.4f}")
    print(f"  multiple regression R2 {multi['r2']:.4f}  adjusted {multi['adj_r2']:.4f}")
    print(f"  log-log adjusted R2 {loglog_adj_r2:.4f}")


if __name__ == "__main__":
    main()
