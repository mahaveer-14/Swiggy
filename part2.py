"""Part 2 - Classification and evaluation.
Run from the repository root:  python part2.py
Needs data/records.csv (from generate_data.py) and common.py.
Writes outputs/part2_results.txt, outputs/part2_thresholds.csv, outputs/part2_knn_cv.csv."""
import math
import os

import numpy as np

from common import (KNN_KS, Results, bce, choose_best_k, confusion, cost_money, f4, knn_cv_scores,
                    knn_positive_counts, knn_predict, metrics, predict_proba, prepare_split,
                    train_logistic)

res = Results()
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------------ Task 2.1
Xtr_raw, Xte_raw, Xtr, Xte, ytr, yte = prepare_split()   # standardised with training mean / SD only

res.put("train_rows", len(ytr))
res.put("train_positives", int(ytr.sum()))
res.put("test_rows", len(yte))
res.put("test_positives", int(yte.sum()))

# ------------------------------------------------------------------ Task 2.2
w, b, losses = train_logistic(Xtr, ytr, eta=0.1, updates=1000, record=(1, 10, 100, 1000))
res.put("loss_epoch_0", losses[0])
res.put("loss_epoch_1", losses[1])
res.put("loss_epoch_10", losses[10])
res.put("loss_epoch_100", losses[100])
res.put("loss_epoch_1000", losses[1000])
for j in range(7):
    res.put(f"weight_X{j + 1}", w[j])
res.put("bias", b)
p_test = predict_proba(Xte, w, b)
p_train = predict_proba(Xtr, w, b)
res.put("test_bce", bce(yte, p_test))

# ------------------------------------------------------------------ Task 2.3
for eta in (0.01, 0.1, 1.0):
    _, _, l200 = train_logistic(Xtr, ytr, eta=eta, updates=200, record=(200,))
    res.put(f"loss_epoch_200_eta_{eta}", l200[200])

# ------------------------------------------------------------------ Task 2.4
for j in range(7):
    res.put(f"odds_ratio_X{j + 1}", math.exp(w[j]))

# ------------------------------------------------------------------ Task 2.5
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, p_test >= tau)
    m = metrics(tp, fp, tn, fn)
    res.put(f"tau_{tau}_tp", tp)
    res.put(f"tau_{tau}_fp", fp)
    res.put(f"tau_{tau}_tn", tn)
    res.put(f"tau_{tau}_fn", fn)
    res.put(f"tau_{tau}_accuracy", m["accuracy"])
    res.put(f"tau_{tau}_precision", m["precision"])
    res.put(f"tau_{tau}_recall", m["recall"])
    res.put(f"tau_{tau}_f1", m["f1"])
    res.put(f"tau_{tau}_specificity", m["specificity"])
res.put("blind_accuracy", (len(yte) - int(yte.sum())) / len(yte))   # predict 0 for everyone

# ------------------------------------------------------------------ Task 2.6
with open(os.path.join("outputs", "part2_thresholds.csv"), "w") as f:
    f.write("tau,tp,fp,tn,fn,tpr,fpr\n")
    for k in range(1, 10):
        tau = k / 10
        tp, fp, tn, fn = confusion(yte, p_test >= tau)
        m = metrics(tp, fp, tn, fn)
        f.write(f"{f4(tau)},{tp},{fp},{tn},{fn},{f4(m['recall'])},{f4(m['fpr'])}\n")

# ------------------------------------------------------------------ Task 2.7
mean_cv = knn_cv_scores(Xtr, ytr)               # 5-fold CV, folds = training position mod 5 (see common.py)
with open(os.path.join("outputs", "part2_knn_cv.csv"), "w") as f:
    f.write("K,mean_cv_f1\n")
    for k in KNN_KS:
        f.write(f"{k},{f4(mean_cv[k])}\n")

best_k = choose_best_k(mean_cv)                 # ties go to the smaller K
res.put("knn_best_k", best_k)

counts_test = knn_positive_counts(Xtr, ytr, Xte, [best_k])[best_k]
tp, fp, tn, fn = confusion(yte, knn_predict(counts_test, best_k))
m = metrics(tp, fp, tn, fn)
res.put("knn_tp", tp)
res.put("knn_fp", fp)
res.put("knn_tn", tn)
res.put("knn_fn", fn)
res.put("knn_accuracy", m["accuracy"])
res.put("knn_precision", m["precision"])
res.put("knn_recall", m["recall"])
res.put("knn_f1", m["f1"])

# ------------------------------------------------------------------ Task 2.8
best_tau, best_cost = None, None
for k in range(1, 20):
    tau = k / 20
    tp, fp, tn, fn = confusion(ytr, p_train >= tau)
    c = cost_money(fp, fn)
    if best_cost is None or c < best_cost:      # strict < keeps the smaller tau on ties
        best_tau, best_cost = tau, c
res.put("cost_tau", best_tau)
res.put("cost_train_at_tau", best_cost)         # whole rupees, so printed as an integer

_, fp, _, fn = confusion(yte, p_test >= best_tau)
res.put("cost_test_at_tau", cost_money(fp, fn))
_, fp, _, fn = confusion(yte, p_test >= 0.5)
res.put("cost_test_at_0.5", cost_money(fp, fn))
res.put("cost_test_flag_nobody", cost_money(0, int(yte.sum())))   # every positive missed

# ------------------------------------------------------------------ write
res.write(os.path.join("outputs", "part2_results.txt"))
print(open(os.path.join("outputs", "part2_results.txt")).read())
