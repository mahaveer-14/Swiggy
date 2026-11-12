Domain: Support

## How to run

Capstone project: Swiggy Support Decision Intelligence (IIT Mandi, Minor in Artificial Intelligence and Data Science). The data is synthetic; nothing here reflects Swiggy's real records, products or policies.

Requirements: Python 3.10 or newer and NumPy only (`numpy==2.2.6`, pinned in `requirements.txt`). No API key, account or dataset download is needed; everything runs offline. Run these commands from the repository root, in this order:

```
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`python generate_data.py` must print `800 10 170 7cb48ef89914049d76954530b6482e04` (rows, columns, escalated records, MD5 of the file). It writes `data/records.csv`. Each part script writes its files into `outputs/`. `part2.py`, `part3.py` and `part4.py` import the shared helper `common.py` (data loading, the stratified split, logistic regression, KNN, metrics); importing it never runs a part. The committed `outputs/` files are exactly what these scripts regenerate on a clean clone.

Repository layout:

```
README.md   requirements.txt   generate_data.py   common.py
data/records.csv
part1.py  part2.py  part3.py  part4.py
outputs/
  part1_results.txt  part2_results.txt  part3_results.txt  part4_results.txt
  part2_thresholds.csv  part2_knn_cv.csv
  part3_segments.csv  part3_hierarchical.csv
  part4_splits.csv  part4_voting.csv  part4_adaboost.csv  part4_architecture.csv
```

## Part 1 findings

**R1.1 Test of the 8-minute planning claim.** H0 (the mean resolution time is 8.0 minutes) is rejected: `z_stat = 2.8930`, which is larger than the critical value 1.96 at the 5% level (two-tailed). The sample mean is 8.9382 minutes and the 95% interval is 8.3026 to 9.5738, which excludes 8.0. A Type I error here would mean concluding that the average resolution time differs from 8 minutes when it really is 8 minutes, so the team would change its staffing plan for no reason. The gap is 8.9382 − 8.0 = 0.9382 minutes (about 56 seconds, roughly 12% above the plan). Statistically it is clear; in practice it is small per ticket but is worth acting on, because it applies to every ticket and so adds up across the whole queue. The spread is large (SD 9.1723 minutes), so individual tickets vary far more than the average does.

**R1.2 Sample size.** `n_required = 1293` tickets are needed to estimate the average to within half a minute at 95% confidence, but only 800 rows are available. With 800 rows the margin of error is about 1.96 × 0.3243 = 0.64 minutes, wider than the 0.5 target. The planning team needs roughly 493 more tickets (1,293 − 800) before it can claim half-minute precision.

**R1.3 Gradient descent versus the closed form.** `gd_slope_gap = 0.0000` (below the 0.001 limit). The squared-error cost is a single bowl-shaped (convex) surface in the slope and intercept, so with a sensible step size gradient descent walks down to its one minimum, which is exactly where the closed-form least-squares line sits (the cost stops changing: 40.3698 after both 100 and 500 updates).

**R1.4 Funnel check.** `resid_sd_ratio_raw = 3.7525`: the residual spread in the upper half of fitted values (8.6891) is almost four times that in the lower half (2.3155), so the raw model shows a clear funnel. `resid_sd_ratio_log = 1.0396`: after taking logs the two halves have nearly equal spread (0.5165 and 0.5369), so the log model removes the funnel.

**R1.5 Multiple regression.** Predictors in Task 1.7 with an absolute coefficient above 0.05 (excluding the constant): ln(X4) (1.1576), X2 (0.0940), X3 (0.1067) and X6 (0.1518). X1 (−0.0074), X5 (0.0028) and X7 (0.0172) fall below the threshold. This is a mechanical rule on coefficients measured in different units, so it is not a ranking of importance. The adjusted R-squared rises from 0.6561 (`loglog_adj_r2`) to 0.7403 (`mr_adj_r2`), so the extra predictors earn their place.

## Part 2 findings

**R2.1 Learning rates.** After 200 updates the losses, lowest to highest, are 0.3703 (η = 1.0), 0.3776 (η = 0.1) and 0.5091 (η = 0.01). A learning rate that is too small (0.01) is still far from converged after 200 updates; a large one (1.0) reached the minimum fastest here, although a still larger step could overshoot.

**R2.2 Weights and odds ratios.** The three predictors with the largest absolute weight are X3 (0.9013), X1 (−0.5554) and X2 (0.5505). The largest odds ratio is X3 at 2.4629: when the share of the resolution-time target already used rises by one training standard deviation, the odds that the ticket escalates are multiplied by about 2.46, other predictors fixed.

**R2.3 Accuracy is a weak score.** `blind_accuracy = 0.7875` (predict "no escalation" for every ticket) is almost the same as the model's τ = 0.5 accuracy of 0.8042, because only about one ticket in five escalates and the blind rule catches none of them. Lowering τ from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 and lowered precision from 0.5455 to 0.4634 (accuracy also fell, to 0.7625).

**R2.4 Logistic versus KNN.** At τ = 0.3 the logistic model has F1 0.5714 and recall 0.7451; the KNN model (K = 5) has F1 0.3696 and recall 0.3333. The logistic model is clearly better on both.

**R2.5 What to deploy.** I would deploy the logistic model with the cost-chosen threshold τ = 0.15. On the test rows it costs INR 6,050 (`cost_test_at_tau`), against INR 11,800 at τ = 0.5 (`cost_test_at_0.5`) and INR 20,400 if nobody is flagged (`cost_test_flag_nobody`). The threshold was chosen on the training rows so that the test rows stay untouched and give an honest estimate of the cost; choosing it on the test rows would make the test cost look better than it really is.

## Part 3 findings

**R3.1 Elbow.** The successive WCSS drops are WCSS(2) − WCSS(3) = 659.8282, WCSS(3) − WCSS(4) = 108.0739, WCSS(4) − WCSS(5) = 171.0575 and WCSS(5) − WCSS(6) = 45.5466. The elbow supports K = 3 (`elbow_k = 3`): the big gain comes from going from 2 to 3 clusters, and the next drop (108.07) is below half of it (659.83 / 2 = 329.9). The drops do not shrink steadily (171.06 comes after 108.07), because the pinned starting rows give one reproducible solution, not necessarily the global optimum.

**R3.2 Segments** (K = 3, ordered by mean X3; from `part3_segments.csv`):

| Segment | Size | Mean X3 | Mean earlier tickets | Mean account value (INR thousands) | Outcome rate | Mean resolution (minutes) |
|---|---|---|---|---|---|---|
| A | 322 | 0.2250 | 1.95 | 12.53 | 0.0186 | 3.60 |
| B | 255 | 0.4534 | 6.07 | 28.72 | 0.2353 | 17.12 |
| C | 223 | 0.7118 | 4.23 | 16.51 | 0.4664 | 7.29 |

- **A, "Fast, low-contact tickets":** little of the time target is used on arrival, few earlier tickets and almost no escalations. Action: keep them in the standard or self-service queue with no supervisor review.
- **B, "Repeat-contact, high-value tickets":** the most earlier tickets, the highest spend and the slowest resolution (17.12 minutes). Action: assign one experienced owner per customer and follow up proactively, since these take the longest.
- **C, "Late-to-agent, likely-to-escalate tickets":** about 71% of the time target is already used when the ticket reaches an agent. Action: fast-track to a senior agent and review before the target is breached.

Segment C has the highest `outcome_rate`, 0.4664.

**R3.3 Hierarchical clustering at the cut.**
- Single linkage gives 4 clusters: {E1}, {E2, E3, E6}, {E4} and {E5}.
- Complete linkage gives 2 clusters: {E1, E4, E5} and {E2, E3, E6}. (Average linkage also gives these two.)

The rules disagree because single linkage measures the distance between the nearest pair of members, so its merge heights stay low (1.3, 1.4, 2.8, 2.9, 3.4) and the largest gap falls early, after the second merge. Complete linkage uses the farthest pair, so the final merge (E1 to E2) happens at 7.9, which creates a huge gap and a cut into two groups.

**R3.4 PCA.** After standardising, the mean vector (step 1) is all zeros. PC1 loads positively on X1 (0.5317, tenure) and X5 (0.4418, satisfaction) and negatively on X3 (−0.5187, share of time target used), X2 (−0.4100, earlier tickets) and X4 (−0.2916, account value), so it contrasts long-standing, satisfied customers with tickets that arrive late or come from repeat contacts. Three components keep at least 80% of the variance (cumulative ratios 0.5227, 0.7593, 0.8616). The mean PC1 score is 1.7773 for segment A, −0.7831 for B and −1.6709 for C, the reverse of the outcome rates, which rise from 0.0186 (A) to 0.2353 (B) to 0.4664 (C), so PC1 is close to a risk axis that separates the segments.

## Part 4 findings

**R4.1 Best split.** S4 (X1 ≤ 24 months) is the best split by both measures (information gain 0.1252, Gini gain 0.0606, ahead of S3 with 0.1046 and 0.0494), because it produces the purest children: 86 of 188 tickets escalate on the yes side (about 46%) but only 33 of 372 on the no side (about 9%). A tree grown until every leaf is pure would memorise noise in the 560 training rows, leaving tiny leaves that do not generalise, so it would overfit. One rule that stops a tree growing too deep is a maximum depth (others are a minimum number of rows per leaf or a minimum gain).

**R4.2 Which model to ship.** I would ship the logistic model, with the cost-chosen threshold of 0.15 from Part 2. At τ = 0.3, the threshold used in the comparison, it has the best F1 (0.5714) with recall 0.7451. The stump has recall 0.7647 but F1 0.5532; KNN has F1 0.3696 and recall 0.3333; the hard vote has F1 0.5344 and recall 0.6863; the soft vote has F1 0.5507 and recall 0.7451. Combining models did not help: both votes score below the logistic model on F1, because the weak KNN pulls them down.

**R4.3 AdaBoost.** After round 1 the two misclassified points (3 and 7) carry 0.25 each, a total weight of 0.5 (`round1_error = 0.2500`, α1 = 0.5493). Round 2 trusts h2 more than h1 because its weighted error is lower (0.1667 against 0.25), which gives it the larger weight (α2 = 0.8047). The two-round ensemble classifies 6 of 8 points correctly (`ensemble_accuracy = 0.7500`). The out-of-bag fraction is 0.3676.

**R4.4 Architecture** (Model A, option 2, input 28 × 28 × 1):

| Layer | Output shape (height × width × depth) | Parameters |
|---|---|---|
| conv1 (5×5, 6 filters, stride 1, padding 2) + ReLU | 28 × 28 × 6 | 156 |
| max-pool 2×2, stride 2 | 14 × 14 × 6 | 0 |
| conv2 (5×5, 16 filters, stride 1, padding 0) + ReLU | 10 × 10 × 16 | 2,416 |
| max-pool 2×2, stride 2 | 5 × 5 × 16 | 0 |
| flatten | 400 | 0 |
| dense1 (32 neurons) + ReLU | 32 | 12,832 |
| dense2 (10 neurons, output) | 10 | 330 |
| **Total** | | **15,734** |

The fully connected option (784 → 16 → 16 → 10) has 13,002 parameters. For a 128 × 128 × 3 input a fully connected layer of 1,024 neurons needs 50,332,672 parameters, against 1,792 for a convolutional layer with 64 filters of size 3 × 3, because convolution reuses the same small filters across the image.

Model B (recurrent network, 64-dimensional word vectors, 128-dimensional hidden state, 16 classes): W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128, with biases b_h (128) and b_y (16), giving `rnn_total = 26768`. This number does not grow when sequences get longer, because the same weights are reused at every time step.

For 10 classes I would use ReLU in the hidden layers and a softmax output with categorical cross-entropy loss.

A very deep plain network can stop learning in its early layers because the gradient is multiplied through many layers whose derivatives are small (for example saturated sigmoids), so it shrinks towards zero before it reaches the first layers. A skip connection adds a layer's input to its output, giving the gradient a direct path back to the early layers so they keep receiving a useful signal.

## Model-choice memo

To: Head of Customer-Support Operations

Recommendation: put the logistic-regression model into production to decide which tickets get a supervisor review before they escalate. Alert threshold: 0.15, meaning flag a ticket when its escalation probability is at least 0.15.

Why: I chose 0.15 on training data using your cost figures, INR 50 per unnecessary review and INR 400 per missed escalation. On 240 held-out tickets it cost INR 6,050. The default cut-off of 0.5 cost INR 11,800, and flagging nobody cost INR 20,400. At a threshold of 0.3 the model catches 74.51% of escalations (recall 0.7451) with F1 0.5714, while the nearest-neighbour model reaches only 0.3696. Combining models did not help; the best vote scored 0.5507.

The main driver is time already used: the riskiest ticket segment (escalation rate 0.4664) reaches an agent with about 71% of the target consumed, so routing those tickets early is the practical first step.

Risk: the evidence comes from synthetic data, one train/test split and only 51 escalated test tickets, so the threshold and costs may shift on real data. A threshold of 0.15 also flags many tickets (118 of the 240 test tickets, 73 of them false alarms), so check reviewer capacity before launch.

Next models: neither the order-photo model (Model A) nor the ticket-text model (Model B) is worth building before this structured model is live; if one is built next, Model B, the main use case, comes first.

## Limitations

- The data is synthetic and generated by a script, so every conclusion describes the generator, not Swiggy's real support operations.
- Everything rests on one fixed 560/240 split with only 51 escalated test tickets; the best K = 5 and the threshold 0.15 are valid for this split only, and no confidence intervals are given for the test metrics.
- The alert threshold depends on rough cost figures (INR 50 per review, INR 400 per missed escalation); different costs would move it.
- K-means uses pinned starting rows, so it gives one reproducible solution that need not be the global optimum, and the elbow drops are not smooth.
- Models A and B were only sized (parameter counts), never trained, so nothing is known about their accuracy.
- The z-test treats tickets as independent and uses the sample SD as the population SD, and resolution times are right-skewed, so the mean alone does not describe a typical ticket.
