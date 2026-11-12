"""Part 4 - Model-choice and architecture memo (numbers only; no model is trained for A or B).
Run from the repository root:  python part4.py
Needs data/records.csv (from generate_data.py) and common.py (shared with Part 2).
Writes outputs/part4_results.txt, part4_splits.csv, part4_voting.csv, part4_adaboost.csv,
part4_architecture.csv. Entropy, Gini, gains, vote rules and boosting are written here by hand."""
import math
import os

import numpy as np

from common import (Results, confusion, f4, knn_cv_scores, choose_best_k, knn_positive_counts,
                    metrics, predict_proba, prepare_split, train_logistic)

res = Results()
os.makedirs("outputs", exist_ok=True)

# the Task 2.1 split and the Task 2.2 / 2.7 models (same code and numbers as Part 2)
Xtr_raw, Xte_raw, Xtr, Xte, ytr, yte = prepare_split()


# ------------------------------------------------------------------ Task 4.1
def entropy(pos, n):
    """-sum p_i log2 p_i over the two classes (0 for a pure or empty node)."""
    if n == 0:
        return 0.0
    total = 0.0
    for c in (pos, n - pos):
        if c > 0:
            p = c / n
            total -= p * math.log2(p)
    return total


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1.0 - (p * p + (1.0 - p) * (1.0 - p))


# columns of the raw matrix: X1 = 0 ... X7 = 6, original units
splits = {
    "S1": lambda A: A[:, 5] == 1,          # X6 = 1  (reopened_flag)
    "S2": lambda A: A[:, 6] == 1,          # X7 = 1  (self_service_first_flag)
    "S3": lambda A: A[:, 2] > 0.5,         # X3 > 0.5 (sla_consumed_ratio)
    "S4": lambda A: A[:, 0] <= 24,         # X1 <= 24 (customer_tenure_months)
}
n_tr = len(ytr)
pos_tr = int(ytr.sum())
root_ent = entropy(pos_tr, n_tr)
root_gin = gini(pos_tr, n_tr)
res.put("root_entropy", root_ent)
res.put("root_gini", root_gin)

split_stats = {}
with open(os.path.join("outputs", "part4_splits.csv"), "w") as f:
    f.write("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain\n")
    for name, rule in splits.items():
        yes = rule(Xtr_raw)
        n_yes, n_no = int(yes.sum()), int((~yes).sum())
        p_yes, p_no = int(ytr[yes].sum()), int(ytr[~yes].sum())
        info_gain = root_ent - (n_yes / n_tr) * entropy(p_yes, n_yes) - (n_no / n_tr) * entropy(p_no, n_no)
        gini_gain = root_gin - (n_yes / n_tr) * gini(p_yes, n_yes) - (n_no / n_tr) * gini(p_no, n_no)
        split_stats[name] = (n_yes, p_yes, n_no, p_no, info_gain, gini_gain)
        f.write(f"{name},{n_yes},{p_yes},{n_no},{p_no},{f4(info_gain)},{f4(gini_gain)}\n")


def best_by(index):
    best = None
    for name in splits:                     # S1..S4 in order; strict > keeps the earlier split on ties
        if best is None or split_stats[name][index] > split_stats[best][index]:
            best = name
    return best


best_info = best_by(4)
best_gini = best_by(5)
res.put("best_split_by_info_gain", best_info)
res.put("best_split_by_gini_gain", best_gini)

# ------------------------------------------------------------------ Task 4.2
n_yes, p_yes, n_no, p_no, _, _ = split_stats[best_info]
stump_rate_yes = p_yes / n_yes              # share of positive TRAINING rows on each side
stump_rate_no = p_no / n_no
res.put("stump_rate_yes", stump_rate_yes)
res.put("stump_rate_no", stump_rate_no)

# logistic model of Task 2.2 (eta = 0.1, 1000 updates) and the KNN of Task 2.7
w, b, _ = train_logistic(Xtr, ytr, eta=0.1, updates=1000)
best_k = choose_best_k(knn_cv_scores(Xtr, ytr))

prob_logistic = predict_proba(Xte, w, b)
knn_counts = knn_positive_counts(Xtr, ytr, Xte, [best_k])[best_k]
prob_knn = knn_counts / best_k              # share of positive neighbours among the K
stump_yes_side = splits[best_info](Xte_raw)
prob_stump = np.where(stump_yes_side, stump_rate_yes, stump_rate_no)

pred_logistic = prob_logistic >= 0.3
pred_knn = knn_counts >= (best_k + 1) // 2
pred_stump = prob_stump >= 0.3
pred_hard = (pred_logistic.astype(int) + pred_knn.astype(int) + pred_stump.astype(int)) >= 2
pred_soft = (prob_logistic + prob_knn + prob_stump) / 3.0 >= 0.3

with open(os.path.join("outputs", "part4_voting.csv"), "w") as f:
    f.write("model,tp,fp,tn,fn,precision,recall,f1\n")
    for name, pred in [("logistic", pred_logistic), ("knn", pred_knn), ("stump", pred_stump),
                       ("hard_vote", pred_hard), ("soft_vote", pred_soft)]:
        tp, fp, tn, fn = confusion(yte, pred)
        m = metrics(tp, fp, tn, fn)
        f.write(f"{name},{tp},{fp},{tn},{fn},{f4(m['precision'])},{f4(m['recall'])},{f4(m['f1'])}\n")

# ------------------------------------------------------------------ Task 4.3  AdaBoost by hand
y = np.array([+1, +1, +1, -1, -1, -1, +1, -1])
h1 = np.array([+1, +1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, +1, +1, -1, +1, -1, +1, -1])

weights = np.full(8, 1.0 / 8.0)
alphas = []
ada_rows = []
for rnd, h in ((1, h1), (2, h2)):
    eps = float(weights[h != y].sum())                      # weighted error
    alpha = 0.5 * math.log((1.0 - eps) / eps)
    unnorm = weights * np.exp(-alpha * y * h)
    after = unnorm / unnorm.sum()
    for i in range(8):
        ada_rows.append((rnd, i + 1, weights[i], unnorm[i], after[i]))
    res.put(f"round{rnd}_error", eps)
    res.put(f"round{rnd}_alpha", alpha)
    alphas.append(alpha)
    weights = after

with open(os.path.join("outputs", "part4_adaboost.csv"), "w") as f:
    f.write("round,point,weight_before,weight_unnormalised,weight_after\n")
    for rnd, point, wb, wu, wa in ada_rows:
        f.write(f"{rnd},{point},{f4(wb)},{f4(wu)},{f4(wa)}\n")

score = alphas[0] * h1 + alphas[1] * h2
ensemble_pred = np.sign(score)                              # a score of exactly 0 counts as wrong
res.put("ensemble_accuracy", float((ensemble_pred == y).sum()) / 8.0)

# ------------------------------------------------------------------ Task 4.4
res.put("oob_fraction", (1.0 - 1.0 / 560.0) ** 560)

# ------------------------------------------------------------------ Task 4.5  parameter counting


def conv_out(size, filt, pad, stride):
    return (size - filt + 2 * pad) // stride + 1


def conv_params(filt, in_ch, n_filters):
    return filt * filt * in_ch * n_filters + n_filters      # one bias per filter


def dense_params(n_in, n_out):
    return n_in * n_out + n_out                             # one bias per neuron


# Model A option 1: fully connected 784 -> 16 -> 16 -> 10
mlp = [("mlp_hidden1", dense_params(784, 16)),
       ("mlp_hidden2", dense_params(16, 16)),
       ("mlp_output", dense_params(16, 10))]

# Model A option 2: convolutional network, shapes tracked layer by layer
s = 28
shapes = [("input", f"{s}x{s}x1")]
s = conv_out(s, 5, 2, 1)
shapes.append(("conv1 (5x5, 6 filters, pad 2) + ReLU", f"{s}x{s}x6"))
s = conv_out(s, 2, 0, 2)
shapes.append(("max-pool 2x2 stride 2", f"{s}x{s}x6"))
s = conv_out(s, 5, 0, 1)
shapes.append(("conv2 (5x5, 16 filters, pad 0) + ReLU", f"{s}x{s}x16"))
s = conv_out(s, 2, 0, 2)
shapes.append(("max-pool 2x2 stride 2", f"{s}x{s}x16"))
flat = s * s * 16
shapes.append(("flatten", f"{flat}"))
cnn = [("cnn_conv1", conv_params(5, 1, 6)),
       ("cnn_conv2", conv_params(5, 6, 16)),
       ("cnn_dense1", dense_params(flat, 32)),
       ("cnn_dense2", dense_params(32, 10))]

with open(os.path.join("outputs", "part4_architecture.csv"), "w") as f:
    f.write("layer,parameters\n")
    for name, count in mlp + cnn:
        f.write(f"{name},{count}\n")
res.put("mlp_total", sum(c for _, c in mlp))
res.put("cnn_total", sum(c for _, c in cnn))

# why convolution scales: 128 x 128 x 3 input
res.put("fc_layer_params", dense_params(128 * 128 * 3, 1024))
res.put("conv_layer_params", conv_params(3, 3, 64))

# Model B: recurrent network, 64-dim word vectors, 128-dim hidden state, 16 classes
d_in, d_h, d_out = 64, 128, 16
res.put("rnn_total", d_h * d_in + d_h * d_h + d_out * d_h + d_h + d_out)   # W_xh, W_hh, W_hy, b_h, b_y

# ------------------------------------------------------------------ write
res.write(os.path.join("outputs", "part4_results.txt"))
print(open(os.path.join("outputs", "part4_results.txt")).read())

# console-only extras to help with the README (not part of any graded file)
print("--- console only ---")
print("KNN K used:", best_k)
print("Model A option 2 shapes and parameters:")
params = {"conv1": cnn[0][1], "conv2": cnn[1][1], "dense1": cnn[2][1], "dense2": cnn[3][1]}
for label, shape in shapes:
    print(f"  {label:42s} {shape}")
print("  parameters:", params)
print("Model B shapes: W_xh", (d_h, d_in), " W_hh", (d_h, d_h), " W_hy", (d_out, d_h))
print("Round-1 normalised weights:", [round(float(v), 4) for v in
      [r[4] for r in ada_rows[:8]]])
