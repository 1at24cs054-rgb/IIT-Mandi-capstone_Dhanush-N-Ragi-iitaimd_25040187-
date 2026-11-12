Domain: E-commerce

# Flipkart E-commerce Decision Intelligence

A customer-analytics study of 800 synthetic Flipkart shopper records, written to answer three
business questions with mathematics rather than with a library call:

1. Is the planning assumption about average order value still right, and how much sample would
   the estimate need?
2. Which customers will stop purchasing in the next 90 days, and what alert threshold is worth
   its cost in Rupees?
3. What kinds of shoppers does the base actually contain, and what would a later image or text
   model cost in parameters?

Everything is implemented from first principles with the Python standard library and NumPy. The
split rule, the standardisation, logistic regression, k-nearest neighbours, k-means, the three
hierarchical linkages and PCA are all hand-written. `numpy.linalg` is used in exactly the two
places the brief allows it: the normal-equation solve in Part 1 and the eigen-decomposition in
Part 3. No ready-made model, metric or clustering routine produces a reported number.

## Repository layout

```
.
├── README.md                      this file
├── requirements.txt               numpy==2.3.5
├── generate_data.py               the provided generator, committed verbatim
├── data/records.csv               generator output, committed
├── common.py                      shared numerical engine imported by every Part
├── part1.py  part2.py  part3.py  part4.py
├── verify_outputs.py              self-check of the committed outputs
└── outputs/                       every results file and CSV
```

`common.py` holds the split, the standardisation, the logistic trainer, the KNN routines and the
metric helpers. Part 4 imports it instead of re-running Part 2, so no Part ever executes another
Part's script.

## How to run

From a clean clone, in this order:

```bash
git clone <this repository>
cd <repository-name>

pip install -r requirements.txt

python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py

python verify_outputs.py
```

`generate_data.py` must print the fingerprint

```
800 10 170 c6f249ba693f1ebeed318d6d5e18de23
```

which is the row count, the column count, the number of churned records and the MD5 of
`data/records.csv`. Every script is deterministic: there is no random seeding anywhere outside
the generator, so a rerun reproduces the committed `outputs/` files exactly. `verify_outputs.py`
is not part of the graded pipeline; it re-reads `outputs/` and asserts that every required key and
CSV file is present, that each printed rate agrees with the confusion cells beside it, and that
the dataset hash still matches.

## Part 1 findings

**R1.1 — the planning claim fails, but the gap is small.** The mean order value is
`mean_value = 8.9382` hundreds of INR (INR 893.82) with `sd_value = 9.1723` and
`se_value = 0.3243`, giving the 95% interval `[8.3026, 9.5738]`. The test of H0: μ = 8.0 returns
`z_stat = 2.8930`, comfortably past the critical value 1.96, so `reject_h0 = yes`. A Type I error
here would mean declaring that the true average order value has drifted away from the INR 800 plan
when it has not, and needlessly re-cutting category budgets on the strength of a sampling
accident. The gap is `8.9382 − 8.0 = 0.9382` column units, about INR 94 per order, or roughly
11.7% above plan. Statistically it is real; practically it is modest — the gap is a tenth of one
standard deviation, so it is worth a plan revision rather than a re-forecast, and it may simply
reflect basket mix rather than a change in customer behaviour.

**R1.2 — 800 rows are not enough for the precision the team wants.** Sizing the sample for a
margin of error of 0.5 column units (INR 50) at 95% confidence gives `n_required = 1293` against
the 800 rows on hand. The planning team is therefore collecting about 62% of the sample it needs:
any statement of the average to within INR 50 is currently unsupported, and the honest interval is
the wider one above, roughly INR 830 to INR 957.

**R1.3 — gradient descent met the closed form.** `gd_slope_gap = 0.0000`, far inside the 0.001
requirement. The cost fell from `gd_J0 = 163.9178` to `gd_J1 = 119.4579`, `gd_J10 = 41.7973` and
`gd_J100 = 40.3698`, then held at `gd_J500 = 40.3698`: the loss surface for a single standardised
predictor is a convex quadratic, and with a learning rate of 0.1 the iteration walks straight down
to its one minimum, which is the same least-squares solution the closed form computes directly.

**R1.4 — there is a funnel, and the log transform removes it.** Residual spread grows steeply with
the fitted value on the raw scale, `resid_sd_ratio_raw = 3.7525` (lower group 2.3155 against upper
group 8.6891). That is exactly the funnel shape that makes the raw model's error bands untrustworthy
for expensive baskets. On the log scale the two halves behave alike, `resid_sd_ratio_log = 1.0396`
(0.5165 against 0.5369), so the log-log model — `loglog_intercept = −2.4370`,
`loglog_slope = 1.5030`, `loglog_r2 = 0.6565`, `loglog_rmse = 0.5268` — is the sounder base for
multiplicative order-value reasoning. The two R² values live on different scales and are not
comparable, as the brief warns.

**R1.5 — the extra predictors earn their place.** Applying the mechanical |coefficient| > 0.05 rule
to the Task 1.7 fit, four predictors qualify: ln(X4) at `1.1576`, X2 (complaints) at `0.0940`, X3
(discount share) at `0.1067` and X6 (cash-on-delivery) at `0.1518`. X1 (tenure, −0.0074), X5
(loyalty, 0.0028) and X7 (plus membership, 0.0172) fall below the line. The adjusted R² rises from
`loglog_adj_r2 = 0.6561` for the one-predictor log model to `mr_adj_r2 = 0.7403` for the eight-column
fit, an improvement of 0.0842 after paying for seven predictors, so the breadth is buying real
explanatory power rather than being soaked up by the penalty.

## Part 2 findings

**R2.1 — the learning rate can stall or sprint.** Ordering the 200-update training losses from
lowest to highest gives η = 1.0 at 0.3703, η = 0.1 at 0.3776 and η = 0.01 at 0.5091. A learning
rate that is too small has not converged after 200 updates and leaves the loss high, so it wastes
compute and understates the model; a large rate reaches a lower loss fastest here, but on a
less well-conditioned problem the same step size would overshoot and diverge, which is why the
stable 0.1 is used for the main fit.

**R2.2 — discounts dominate the churn signal.** The three largest absolute weights are X3
(discount share) at `0.9013`, X1 (tenure) at `−0.5554` and X2 (complaints) at `0.5505`. The largest
odds ratio is `odds_ratio_X3 = 2.4629`: one training standard deviation more discount-hunting
multiplies the odds of churn by about 2.46, other predictors held fixed — the shoppers who wait for
a coupon are the ones most likely to walk away when the coupon stops.

**R2.3 — accuracy is a trap here.** `blind_accuracy = 0.7875` is the score of a model that predicts
"stays" for everyone, and the fitted model at τ = 0.5 reaches only `tau_0.5_accuracy = 0.8042`,
about 1.7 percentage points better. With 78.75% of the test rows negative, a classifier can look
respectable while catching fewer than half the leavers, so accuracy alone is a weak score. Lowering
the threshold from 0.5 to 0.3 raised recall from `tau_0.5_recall = 0.4706` to `tau_0.3_recall = 0.7451`
(an extra 14 churners flagged) at the cost of precision falling from `tau_0.5_precision = 0.5455` to
`tau_0.3_precision = 0.4634`, and F1 improved from 0.5053 to 0.5714.

**R2.4 — logistic beats KNN on both counts.** At τ = 0.3 the logistic model scores
`tau_0.3_f1 = 0.5714` and `tau_0.3_recall = 0.7451`, while KNN with the cross-validated
`knn_best_k = 5` scores `knn_f1 = 0.3696` and `knn_recall = 0.3333`. KNN is also the more expensive
model to serve, since it stores the whole reference set, so there is no axis on which it wins.

**R2.5 — the money prefers a low threshold.** On the training rows the cost rule picks
`cost_tau = 0.15`, where `cost_train_at_tau = 12150.0000`. Carried to the test rows that threshold
costs `cost_test_at_tau = 6050.0000`, against `cost_test_at_0.5 = 11800.0000` at the textbook 0.5
and `cost_test_flag_nobody = 20400.0000` for doing nothing — so shipping logistic regression at
0.15 would save about INR 14,350 on this test set versus inaction, and about INR 5,750 versus the
default 0.5. The threshold was chosen on training rows precisely so that the test rows stay a clean
witness: picking the threshold on the same 240 customers you then report would flatter the result
and would not survive the next cohort.

## Part 3 findings

**R3.1 — the elbow points at three.** The successive drops are WCSS(2) − WCSS(3) = 659.8282,
WCSS(3) − WCSS(4) = 108.0739, WCSS(4) − WCSS(5) = 171.0575 and WCSS(5) − WCSS(6) = 45.5466. At
K = 3 the next drop, 108.0739, is far below half of the previous one (329.9141), so the rule
returns `elbow_k = 3`. The drops do not shrink steadily — 4 → 5 is larger than 3 → 4 — because the
pinned starting rows give one reproducible solution and not necessarily the global optimum for
every K; other seeds reach lower WCSS at K = 4 and K = 6. The elbow is a heuristic on a fixed
reproducible path, not proof that three is the true number of shopper types.

**R3.2 — three shopper types, three different plays.** Segment A (322 customers, churn
`0.0186`) is the **premium loyal** group: 47.5 months of tenure, a discount share of just 0.225,
loyalty 62.2 and an average order value of 12.53. They are already staying, so the action is
retention hygiene, not discounting — early access and service recovery, with zero coupon spend.
Segment B (255, churn `0.2353`) is the **high-value full-basket** group: the largest average past
order value at 28.72 and the largest mean order value at 17.12, with a middling discount share of
0.453. The action is to trade discount for basket: free delivery and bundle upgrades ahead of a
percentage-off coupon. Segment C (223, churn `0.4664`) is the **bargain-focused newcomer** group:
0.712 discount share, only 14.4 months of tenure, the lowest loyalty at 38.5, and it carries the
highest `outcome_rate` of the three at `0.4664`. The action is a time-boxed win-back on first repeat
purchase rather than a permanent discount, which is what trained the behaviour in the first place.

**R3.3 — the two linkages disagree, and the table says why.** Single linkage cuts into four
clusters: E1, E2+E3+E6, E4 and E5. Complete linkage cuts into two: E1+E4+E5 and E2+E3+E6. The
reason is the shape of the table. Single linkage only ever needs one close pair to merge two groups,
so the tight 2.8 distance between E4 and E5, and the 2.9 between E1 and that pair, chain the whole
E1/E4/E5 block together early and its largest height gap (1.4, between merge 2 and merge 3) sits
inside the E2/E3/E6 side, splitting it off. Complete linkage charges a merge by its worst
member-to-member distance, so it refuses to join E1 to E4+E5 until height 7.9, which makes that
single jump the largest gap in its profile and collapses the table to two blocks.

**R3.4 — one component carries the shopper story.** PC1 is a contrast: positive on X1 tenure
(`pc1_loading_X1 = 0.5317`) and X5 loyalty (`0.4418`), negative on X3 discount share
(`−0.5187`), X2 complaints (`−0.4100`) and X4 average order value (`−0.2916`). It separates settled,
loyal, full-price customers from short-tenure, complaint-heavy, discount-driven ones.
`components_for_80pct = 3`, since PC1 alone explains `explained_ratio_1 = 0.5227` and the first
two reach `cumulative_ratio_2 = 0.7593`, so a third component is needed before 80% is passed. The
segment scores follow the same axis: `mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = −0.7831`
and `mean_pc1_segment_C = −1.6709`, so A sits at the loyal end, C at the discount-driven end, and
B in between — which is a useful confirmation that the K = 3 clustering and the first principal
component are telling the same story rather than two different ones.

## Part 4 findings

**R4.1 — the tenure split wins because it separates the base.** On the 560 training rows
`root_entropy = 0.7462` and `root_gini = 0.3347`. The four candidate splits gain
`S1 0.0110`, `S2 0.0033`, `S3 0.1046` and `S4 0.1252`, with Gini gain ranking them identically, so
`best_split_by_info_gain = S4` (tenure ≤ 24 months) and `best_split_by_gini_gain = S4`. It wins
because it puts 86 of 188 churners on the "yes" side — a 45.74% churn rate against 8.87% on the
other side — whereas the two flag columns barely move the class mix at all. A tree grown until
every leaf is pure would keep splitting on noise until each leaf held one or two customers, which
describes the training rows and nothing else; limiting the depth (or requiring a minimum number of
rows per leaf) is the usual brake.

**R4.2 — ship the single logistic model; the ensemble does not pay.** At τ = 0.3 the logistic model
reaches `f1 0.5714` and `recall 0.7451`. Combining it with KNN and the stump did not help:
`hard_vote` scores `f1 0.5344` and `recall 0.6863`, and `soft_vote` scores `f1 0.5507` and
`recall 0.7451`. Both votes drag F1 below the single model because the KNN member is weak
(`f1 0.3696`) and a majority or mean rule inherits its errors without gaining anything the logistic
model did not already have. The stump alone (`f1 0.5532`, `recall 0.7647`) is the closest
competitor and is nearly free to serve, but it still trails on F1.

**R4.3 — round 1 leaves half the weight on the two errors.** `round1_error = 0.2500` and
`round1_alpha = 0.5493`; after normalisation the two points h1 gets wrong (points 3 and 7) hold
`0.5000` of the total weight between them, up from 0.2500 at the start, which is what forces round 2
to look at them. Round 2 then trusts h2 more (`round2_alpha = 0.8047` against 0.5493) because h2's
weighted error falls to `round2_error = 0.1667`. The two-rule ensemble classifies
`ensemble_accuracy = 0.7500` of the eight points — which is exactly what h1 alone scores. It repairs
both of h1's misses (points 3 and 7, where h2 is right) but breaks two points h1 had correct
(points 1 and 5), because on every disputed point the two rules disagree and the larger weight
α2 > α1 hands h2 the decision. Boosting helped the weighted error, not the unweighted accuracy:
two rounds were not enough for the ensemble to overtake its strongest member.

**R4.4 — what the later models would need.** Model A option 1 (784 → 16 → 16 → 10) needs
`mlp_total = 13002` parameters; option 2, the convolutional network, needs `cnn_total = 15734`,
built layer by layer:

| Layer | Output shape (h × w × d) | Parameters |
|---|---|---|
| conv1, 5 × 5 × 6, pad 2, stride 1 | 28 × 28 × 6 | 156 |
| max-pool 2 × 2, stride 2 | 14 × 14 × 6 | 0 |
| conv2, 5 × 5 × 16, pad 0, stride 1 | 10 × 10 × 16 | 2416 |
| max-pool 2 × 2, stride 2 | 5 × 5 × 16 | 0 |
| flatten | 400 | 0 |
| dense, 32 ReLU | 32 | 12832 |
| dense, 10 outputs | 10 | 330 |

The convolutional model carries slightly more parameters yet shares its 5 × 5 filters across all
784 input positions, which is why the comparison further down matters: a fully connected layer from
a 128 × 128 × 3 input to 1,024 neurons would need `fc_layer_params = 50332672` parameters, while a
convolutional layer with the same input and 64 filters of 3 × 3 needs only
`conv_layer_params = 1792` — 28,000 times fewer, because the filter is one small set of weights
reused at every location. Model B's recurrent network has `W_xh` of shape 64 × 128 (8,192 weights),
`W_hh` of shape 128 × 128 (16,384) and `W_hy` of shape 128 × 16 (2,048), plus 128 hidden biases and
16 output biases, for `rnn_total = 26768`; that number is fixed by the word-vector size, the hidden
size and the class count, so it does not grow at all when the review gets longer — only the number
of times the same matrices are applied grows. For the 10-class picture model I would use ReLU in
the hidden layers and a softmax output trained with categorical cross-entropy. A very deep plain
network can stop learning in its early layers because the gradient is repeatedly multiplied by small
derivatives as it travels back, shrinking towards zero; a skip connection adds an identity path that
carries the gradient straight through, so the early layers keep receiving a usable signal.

## Model-choice memo

To: Head of Retention, Customer Analytics

Recommendation: put **logistic regression** into production at an alert threshold of **0.15**, and
nothing more elaborate.

The evidence. On the 240 held-out customers this model at the money-optimal threshold costs
**INR 6,050**. The same model at the textbook 0.50 threshold costs **INR 11,800**, and flagging
nobody costs **INR 20,400** — so acting at 0.15 is worth about INR 14,350 of avoided loss against
inaction, and about INR 5,750 against the default cut-off. That ordering is what the coupon
economics imply: a false alarm wastes INR 50, a missed churner costs INR 400, so the model should
lean towards alerting. Every one of those figures comes from the training/validation protocol in
Part 2, and the 0.15 threshold was selected on the **training** rows, where it cost
`cost_train_at_tau = 12,150`, so the test rows remain an honest witness.

I would not ship an ensemble. The hard vote and the soft vote both scored lower F1 than the single
logistic model on the same test rows, and the k-nearest-neighbour member was too weak to add
anything.

The main risk: only 51 churners sit in the test set, so a single flipped customer moves the cost by
INR 400 and the threshold estimate is fragile. Re-estimate 0.15 on a rolling window before it is
hard-coded.

Model A or Model B first? Neither — both are worth building only once the structured model is live
and has been re-fit on several cohorts; the churn problem is not constrained by model capacity.

## Limitations

1. **The data is synthetic.** All 800 rows come from the provided generator, with three hidden
   groups injected by construction. So the clustering in Part 3 partly recovers the generator's own
   design rather than discovering genuinely unknown shopper types, and the clean segment separation
   on PC1 should be read as a check that the pipeline works, not as a finding about real customers.
   The generator also builds churn with a fixed logistic form, which flatters a logistic classifier.
2. **One split, and few positives.** Everything in Parts 2 and 4 rests on a single stratified 70/30
   split, and the test set holds only 51 churners. Part 2 chose K for KNN and the alert threshold by
   cross-validation and training-set search respectively, but the reported test metrics still carry
   the sampling noise of those 51 rows; a different split would move accuracy and F1 by a few points
   and could move the chosen K.
3. **The threshold is pinned to one cost ratio.** The 0.15 cut-off is optimal only for the assumed
   INR 50 / INR 400 economics. If a coupon were much cheaper or a lost customer much less valuable,
   the optimal threshold would move, so the recommendation should be recomputed rather than reused
   whenever campaign costs change.
4. **Part 3's elbow is a heuristic on a pinned path.** The K-means result depends on the pinned
   starting centroids; other seeds reach a lower WCSS for some K, so the elbow reflects this
   reproducible path and not a unique global optimum. The hierarchical cut is likewise read from a
   largest-gap rule, which is a convention rather than a statistical test.
5. **No causal claim.** Part 1 shows that average order value differs from the INR 800 plan and
   Part 2 shows which customers are associated with churn; neither shows that a discount, a
   complaint or a coupon caused the behaviour. Targeting segment C with a win-back offer is a
   hypothesis to test in an experiment, not a result of this analysis.
