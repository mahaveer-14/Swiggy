"""Part 1 - Estimation, inference and a regression baseline.
Run from the repository root:  python part1.py
Reads data/records.csv (made by generate_data.py), writes outputs/part1_results.txt.
Only the standard library and NumPy are used; every statistic is computed by the
formulas in the brief (no polyfit / cov / corrcoef / lstsq)."""
import csv
import math
import os

import numpy as np

# ------------------------------------------------------------------ helpers
results = []  # list of (key, formatted value) in the order written


def put(key, value):
    """Format a value as the brief requires and remember it."""
    if isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, str):
        text = value
    else:
        text = format(float(value), ".4f")
    results.append((key, text))


def sample_var(a):
    """Sample variance, divisor n - 1."""
    d = a - a.mean()
    return float((d * d).sum() / (len(a) - 1))


def sample_sd(a):
    return math.sqrt(sample_var(a))


def sample_cov(a, b):
    return float(((a - a.mean()) * (b - b.mean())).sum() / (len(a) - 1))


def simple_ols(x, y):
    """Closed-form one-predictor regression: slope = cov/var, then intercept."""
    slope = sample_cov(x, y) / sample_var(x)
    intercept = float(y.mean()) - slope * float(x.mean())
    return slope, intercept


def split_sd_ratio(fitted, resid):
    """Sort by fitted value (stable, ties by row index); SD of residuals in the
    first 400 and last 400 rows; return (lower, upper, upper / lower)."""
    order = np.argsort(fitted, kind="stable")
    half = len(order) // 2
    lower = sample_sd(resid[order[:half]])
    upper = sample_sd(resid[order[-half:]])
    return lower, upper, upper / lower


# ------------------------------------------------------------------ load data
with open(os.path.join("data", "records.csv")) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

col = {name: np.array([float(r[i]) for r in rows]) for i, name in enumerate(header) if i > 0}
X1 = col["customer_tenure_months"]
X2 = col["earlier_ticket_count_90d"]
X3 = col["sla_consumed_ratio"]
X4 = col["account_value_inr_thousands"]
X5 = col["satisfaction_score"]
X6 = col["reopened_flag"]
X7 = col["self_service_first_flag"]
Y = col["resolution_minutes"]          # the "value" column
n = len(Y)

# ------------------------------------------------------------------ Task 1.1
mean_value = float(Y.mean())
sd_value = sample_sd(Y)
se_value = sd_value / math.sqrt(n)
put("n", n)
put("mean_value", mean_value)
put("sd_value", sd_value)
put("se_value", se_value)
put("ci95_low", mean_value - 1.96 * se_value)
put("ci95_high", mean_value + 1.96 * se_value)

# ------------------------------------------------------------------ Task 1.2
z_stat = (mean_value - 8.0) / se_value
put("z_stat", z_stat)
put("reject_h0", abs(z_stat) > 1.96)

# ------------------------------------------------------------------ Task 1.3
E = 0.5
n_required = math.ceil((1.96 * sd_value / E) ** 2)   # smallest integer n >= (1.96 sd / E)^2
put("n_required", n_required)

# ------------------------------------------------------------------ Task 1.4
reg_slope, reg_intercept = simple_ols(X4, Y)
fitted = reg_intercept + reg_slope * X4
resid = Y - fitted
ybar = float(Y.mean())
tss = float(((Y - ybar) ** 2).sum())
ess = float(((fitted - ybar) ** 2).sum())
rss = float((resid ** 2).sum())
put("reg_slope", reg_slope)
put("reg_intercept", reg_intercept)
put("tss", tss)
put("ess", ess)
put("rss", rss)
put("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
put("r2", 1.0 - rss / tss)
put("rmse", math.sqrt(rss / n))
put("mae", float(np.abs(resid).mean()))

# ------------------------------------------------------------------ Task 1.5
mean_x4 = float(X4.mean())
sd_x4 = sample_sd(X4)
z = (X4 - mean_x4) / sd_x4


def cost(m, c):
    e = Y - (m * z + c)
    return float((e * e).sum() / n)


m, c = 0.0, 0.0
eta = 0.1
put("gd_J0", cost(m, c))
saved = {}
for step in range(1, 501):
    e = Y - (m * z + c)
    grad_m = -(2.0 / n) * float((z * e).sum())
    grad_c = -(2.0 / n) * float(e.sum())
    m -= eta * grad_m
    c -= eta * grad_c
    if step in (1, 10, 100, 500):
        saved[step] = cost(m, c)
put("gd_J1", saved[1])
put("gd_J10", saved[10])
put("gd_J100", saved[100])
put("gd_J500", saved[500])
gd_slope = m / sd_x4
gd_intercept = c - m * mean_x4 / sd_x4
put("gd_slope", gd_slope)
put("gd_intercept", gd_intercept)
put("gd_slope_gap", abs(gd_slope - reg_slope))

# ------------------------------------------------------------------ Task 1.6
lo_raw, up_raw, ratio_raw = split_sd_ratio(fitted, resid)
put("resid_sd_lower_raw", lo_raw)
put("resid_sd_upper_raw", up_raw)
put("resid_sd_ratio_raw", ratio_raw)

ln_y = np.log(Y)
ln_x4 = np.log(X4)
ll_slope, ll_intercept = simple_ols(ln_x4, ln_y)
ll_fitted = ll_intercept + ll_slope * ln_x4
ll_resid = ln_y - ll_fitted
ll_tss = float(((ln_y - ln_y.mean()) ** 2).sum())
ll_rss = float((ll_resid ** 2).sum())
ll_r2 = 1.0 - ll_rss / ll_tss
put("loglog_intercept", ll_intercept)
put("loglog_slope", ll_slope)
put("loglog_r2", ll_r2)
put("loglog_rmse", math.sqrt(ll_rss / n))
lo_log, up_log, ratio_log = split_sd_ratio(ll_fitted, ll_resid)
put("resid_sd_lower_log", lo_log)
put("resid_sd_upper_log", up_log)
put("resid_sd_ratio_log", ratio_log)

# ------------------------------------------------------------------ Task 1.7
# columns: constant, ln(X4), X1, X2, X3, X5, X6, X7  (exact order)
design = np.column_stack([np.ones(n), ln_x4, X1, X2, X3, X5, X6, X7])
beta = np.linalg.solve(design.T @ design, design.T @ ln_y)   # normal equation
for j in range(8):
    put(f"mr_beta_{j}", beta[j])
mr_resid = ln_y - design @ beta
mr_rss = float((mr_resid ** 2).sum())
mr_r2 = 1.0 - mr_rss / ll_tss                 # same TSS: both models explain ln(value)
p = 7
put("mr_r2", mr_r2)
put("mr_adj_r2", 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1))
put("loglog_adj_r2", 1.0 - (1.0 - ll_r2) * (n - 1) / (n - 1 - 1))

# ------------------------------------------------------------------ write file
os.makedirs("outputs", exist_ok=True)
with open(os.path.join("outputs", "part1_results.txt"), "w") as f:
    for key, text in results:
        f.write(f"{key} = {text}\n")
print(open(os.path.join("outputs", "part1_results.txt")).read())
