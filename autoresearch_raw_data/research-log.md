# Research log: autoresearch_raw_data run1

Metric: mean AUC-ROC over 4 time-ordered expanding folds (prepare.run_cv). Fold noise ~0.01.

## Setup and EDA (raw_train.csv only)

- 4,035 customers, 99,830 lines, all customers have exactly one invoice; no negative quantities, no zero prices.
  Only 16 non-numeric stock codes (C2 = carriage). Dates 2009-12-01 to 2010-11-30.
- Churn rate 44.7%, drifting upward over time: 33% (Dec-09 cohort) to 55% (Nov-10). Folds validate on the
  latest cohorts, so a model must extrapolate in time.
- 92.5% UK (churn 45%); Germany 28%, France 15%, Channel Islands 9%: non-UK customers churn less.
- Hour of day and weekday show mild differences (Sat almost absent).

## Web research before experiment 2

- Kopczewska et al., "Churn in retail e-commerce after the first purchase" (summary at
  https://encyclopedia.pub/entry/18374): XGBoost best when basic features are combined with dummies for the
  product categories bought on the first purchase -> product-level encodings of the basket.
- Online Retail II churn project (https://daffadaniel.github.io/05-ChurnModelling/): RFM-style features,
  test AUC ~0.62 -> signal is weak; expect modest gains, prefer regularised shallow trees.
- Repeat-buyer prediction literature (https://univagora.ro/jour/index.php/ijccc/article/view/4988,
  https://www.netelixir.com/?p=10984): basket size, order value and product diversity of the first order.

Plan: (1) regularise XGBoost for a weak signal; (2) richer basket stats; (3) country; (4) time-of-purchase;
(5) product-level out-of-fold target encoding (the "category dummies" idea, compressed); (6) description keywords.

## Experiments

### 1 - baseline (keep, 0.5947)
Unmodified train.py: 5 roll-ups, default-ish XGBoost (100 trees, depth 6, lr 0.1). Folds 0.601/0.591/0.580/0.608.

### 2 - regularised XGBoost (exploration; keep, 0.5999)
Hypothesis: weak signal + 5 features -> depth 6 overfits; shallow trees, lower lr, subsampling help (web: AUC ~0.62
ceiling in similar setups). Result +0.005, all four folds up or flat (0.601/0.592/0.591/0.616).

### 3 - basket distribution stats (exploration; keep, 0.6099)
Hypothesis (repeat-buyer literature: cart size/value/diversity): shape of the basket (wholesale-like large
quantities, price range) distinguishes resellers who come back from one-off gift buyers. +0.010, all folds up
(0.607/0.600/0.600/0.632).

### 4 - is_uk flag (exploration; discard, 0.6084)
Hypothesis (EDA): non-UK customers churn far less (Germany 28%, France 15%). Result -0.0015 (folds
0.603/0.595/0.603/0.633): mixed, folds 1-2 down. Only ~7% non-UK, so few validation customers benefit; the
binary flag adds little the basket features do not already capture (foreign customers buy larger baskets).

### 5 - hour + weekday of first invoice (exploration; discard, 0.5973)
Hypothesis (EDA): evening/late purchases churn less; Monday more. Result -0.013, all folds down: these are mostly
noise splits that the trees chase. Time-of-day is not useful here.

### 6 - product out-of-fold target encoding (exploration)
Hypothesis (Kopczewska et al.: product categories of first basket are the strongest add-on): each stock_code's
smoothed churn rate among fit customers (5-fold out-of-fold for fit rows to avoid self-leakage, full fit stats
for apply rows, smoothing m=20 towards the prior), aggregated per customer as mean/min/max, plus the average
log popularity of the basket's products among fit customers.
Result: discard, 0.6020 (0.599/0.600/0.601/0.609). Fold 4 dropped most (-0.023). Likely causes: products are
seasonal (Christmas stock in early cohorts vs. summer stock later) and churn drifts upward over time, so past
product churn rates do not transfer; 4 noisy TE columns also dilute the 300-tree budget.

### Web research after 3 consecutive discards (before 7)
- UCI Online Retail II page (https://archive.ics.uci.edu/dataset/502/online+retail+ii): "many customers of the
  company are wholesalers" -> wholesale behaviour must be inferred from order shape (pack quantities).
- Wholesale vendor note (https://wholesalesuiteplugin.com/?p=946): wholesale buyers reorder more readily.
- Pargent et al. 2022 (https://link.springer.com/article/10.1007/s00180-022-01207-6) and arXiv 2601.10019
  (https://arxiv.org/html/2601.10019v1): regularised (strongly smoothed, OOF) TE, and time-aware TE under drift.
  -> if TE is retried, use stronger smoothing / fewer columns / recency-weighted stats.

### 7 - wholesale pack-pattern features (follow-up on 3)
Hypothesis (UCI note): resellers buy in pack multiples (6/12/24...) and rarely single units. Add share of lines
with quantity a multiple of 12, share of lines with quantity >= 12, share of single-unit lines.
Result: discard, 0.6068 (0.600/0.598/0.599/0.630). Folds 1-3 slightly down: median/max quantity already encode
this. Four discards in a row: every added feature hurts a little -> the model may be capacity/variance-limited
rather than feature-limited. Switch to tuning the model on the 11 kept features.

### 8 - depth 2, 500 trees (exploration, model)
Hypothesis: with weak signal, depth-2 trees (pairwise interactions only) generalise better across cohorts.
Result: discard, 0.6091 (0.604/0.600/0.605/0.628): folds 1 and 3 up, fold 4 down; within noise of best.
Depth 2 vs 3 is not the lever.

### 9 - Christmas-product share (exploration, text)
Hypothesis: the earliest cohorts churn least and are Christmas-heavy, but within a cohort a Christmas-themed
basket marks a seasonal gift buyer (one-off) vs. a year-round reseller. Add share of lines whose description
contains CHRISTMAS.
Result: discard, 0.6084 (0.603/0.595/0.606/0.630). Fold 3 up, fold 2 down: noise-level.

### Web research after 3 consecutive discards (7-9, before 10)
- XGBoost parameter docs (https://xgboost.readthedocs.io/en/release_1.6.0/parameter.html): min_child_weight is a
  hessian sum; for logloss (hessian <= 0.25) a leaf needs ~4x that many rows, so 5 means only ~20 customers.
- MetricGate tuning guide (https://metricgate.com/blogs/xgboost-hyperparameter-tuning-guide/): on noisy data
  default lambda=1/gamma=0 is often suboptimal; tune structure, then randomness, then regularisation, then lr.

### 10 - min_child_weight 5 -> 20 (follow-up on research)
Hypothesis: leaves of ~80+ customers average out label noise; currently ~20-customer leaves overfit.
Result: discard, 0.6051 (0.602/0.598/0.595/0.626), all folds down. More conservative leaves lose signal
rather than noise: the useful splits are on small groups (e.g. large wholesale baskets).

## Synthesis after 10 experiments
- Works: regularised but not over-regularised XGBoost (depth 3, lr 0.03, 300 trees, mcw 5); basket-shape stats
  (quantity median/max, price min/max/std, max line total). Best 0.6099.
- Does not work: is_uk (-0.002), hour/weekday (-0.013), product TE (-0.008), pack-pattern shares (-0.003),
  Christmas share (-0.002), depth 2 (-0.001), mcw 20 (-0.005).
- Pattern: seven discards in a row, most within ~0.003 = noise floor. Basket-shape is saturated; the next real
  gain must come from a genuinely new information source (what products are bought, not how many) represented
  compactly, or from feature selection (fewer, stronger features).

## Web research at experiment 10 (before 11)
- MDPI AI 7(244) retail e-commerce churn (https://mdpi-res.com/d_attachment/ai/ai-07-00244/article_deploy/ai-07-00244.pdf):
  product descriptions are a usable field; best results came from feature selection + XGBoost.
- Groovyweb churn guide (https://www.groovyweb.co/blog/predict-customer-churn-machine-learning): churn is
  mainly a feature-engineering problem; GBMs win via non-linear interactions.
- No direct source for TF-IDF+SVD basket features; testing it as an unsupervised, drift-robust alternative to
  product TE (no labels used, so no label drift).

### 11 - TF-IDF + SVD basket text topics (exploration)
Hypothesis: a customer's basket text (all descriptions) projected onto 8 latent topics (TfidfVectorizer +
TruncatedSVD fitted on fit customers only) captures what kind of shop the customer is (gift/party vs. kitchen
vs. home decor) without using labels.
Result: discard, 0.5945 (0.604/0.591/0.591/0.592); fold 4 collapsed (-0.040). Like product TE (exp 6), what
products are bought does not transfer across time: the assortment is seasonal and the topic axes are fitted on
a different season. Conclusion: avoid product-identity features; stick to how/what-price the customer buys.

### 12 - relative price paid vs. product's typical price (exploration)
Hypothesis: this retailer has volume price tiers, so paying below a product's median unit price (median over
fit lines, label-free) marks a bulk/trade buyer. Add mean and min of price / product median price (unknown
products -> 1.0). Uses product identity only as a price reference, which should be stable over time.
Result: discard, 0.6041 (0.599/0.597/0.602/0.619). Not useful; price tiers are already reflected in
min_unit_price and quantities.

### Web research after 3 consecutive discards (10-12, before 13)
- Adversarial validation (https://hamel.dev/blog/posts/drift/, https://www.kdnuggets.com/2016/10/adversarial-validation-explained.html,
  https://arxiv.org/pdf/2004.03045v1.pdf): train/predict early-vs-late membership to find drifting features.
- Ran a label-free check (helper drift_check.py; early 60% vs late 40% customers, single-feature AUC): all 11
  kept features have adversarial AUC within 0.43-0.54 -> basket features barely drift. The discards are not a
  drift problem for basket features (they were for product content, exps 6/11); the issue is variance of a
  weak-signal model.

### 13 - smoother ensemble: lr 0.03 -> 0.015, 300 -> 600 trees (model)
Hypothesis: same total shrinkage, more and smaller steps + row/col subsampling = more averaging, less variance.
Result: discard, 0.6084 (0.606/0.599/0.603/0.626): folds 1-3 slightly up, fold 4 down. Within noise.

### 14 - basket concentration ratios (exploration)
Hypothesis: shape ratios the trees cannot easily form from sums: mean quantity per line, spend per line, and
top-line share (max_line_total / total_spend; a basket dominated by one bulk line = reseller restocking).
Result: discard, 0.6066 (0.602/0.596/0.598/0.630). Again every added column costs ~0.003 on folds 1-3.

### Web research after 3 consecutive discards (12-14, before 15)
- ASOS CLV model (Chamberlain et al., https://arxiv.org/pdf/1703.02596): customer nationality is among the top
  features -> country deserves a richer encoding than a UK flag (try later).
- TUM early-CLV work (https://mediatum.ub.tum.de/1156083): early-relationship signals are predictive but weak.
- Pattern in my own log: additions cost ~0.003 each -> try the opposite direction, removing redundant columns.

### 15 - drop n_lines (ablation)
Hypothesis: with one invoice per customer, n_lines ~= n_distinct_products; the duplicate wastes colsample
draws. Removing it should be neutral or positive and simpler.
Result: discard, 0.6091 (0.605/0.600/0.605/0.627): folds 1-3 UP vs best (0.607/0.600/0.600), fold 4 down 0.005.
Fold 4 is the smallest validation window (~10% of customers) and the noisiest; the best's 0.632 there looks
partly lucky. Strict rule says discard.

### 16 - country out-of-fold target encoding (follow-up on research, ASOS)
Hypothesis: nationality matters (ASOS CLV; EDA: Germany 28%, France 15%, Channel Islands 9% churn vs UK 45%).
Smoothed (m=30) churn rate per country, 5-fold OOF for fit customers, full fit stats for apply customers.
Result: discard, 0.6068 (0.606/0.597/0.595/0.630). Country is confirmed useless here (two encodings tried):
too few non-UK customers per validation window to move AUC.

### 17 - no row/column subsampling (model, variance)
Hypothesis: with 11 features colsample 0.8 drops 2 features per tree and subsample adds noise; a deterministic
fit (subsample=1, colsample=1) may be more stable. Also tests whether 0.6099 depends on the random draw.
Result: discard, 0.6091 (0.603/0.604/0.600/0.629). Again 0.609x. Five different variants now land at
0.6084-0.6091: the true level of this feature set is ~0.609; 0.6099 is the top of the noise band.

### Web research after 3 consecutive discards (15-17, before 18)
- Google Research, instance-conditional decay (https://research.google/blog/learning-the-importance-of-training-data-under-concept-drift/)
  and Scholz & Klinkenberg, boosting for drifting concepts (https://scispace.com/papers/boosting-classifiers-for-drifting-concepts-26f20voi82):
  under drift, re-weight recent training data. Not applicable cleanly: run_cv calls model.fit(X, y) with no
  sample_weight, and wrapping XGBClassifier to smuggle weights would be a hack. Noted, not pursued.

### 18 - stock-code variant buying (exploration)
Hypothesis: codes like 79323P / 79323W are colour/size variants of a base item. Buying several variants of the
same base item = stocking a range for resale (returns), vs. a single pick (consumer). Label-free and not tied
to a season. Add n_variant_groups (base codes bought in >1 variant) and share_variant_lines.
Result: discard, 0.6063 (0.608/0.592/0.599/0.626). Fold 1 best ever, fold 2 down. Noise-level.

### 19 - depth 4, lr 0.02 (model)
Hypothesis: the opposite of exp 8/10: maybe 3-way+ interactions (e.g. large basket x cheap items x high max
quantity) carry signal; slower lr keeps it from overfitting.
Result: discard, 0.6082 (0.605/0.596/0.601/0.631). Depth 2/3/4 all give ~0.609: model shape is not the lever.

### 20 - left-censoring flag for the first month of data (exploration, de-confounding)
Hypothesis: data begins 2009-12-01, so the Dec-2009 "first invoices" include existing customers seen for the
first time (EDA: churn 33% vs 40-55% later). They distort the basket->churn mapping. A flag
`in_first_month` (first invoice within 31 days of the earliest fit date) lets the trees route those customers
separately; for validation customers it is always 0, so they are scored with the "genuinely new" mapping.
Result: discard, 0.6063 (0.603/0.598/0.600/0.625). The basket->churn mapping is evidently similar for the
Dec-09 cohort; the flag only adds a split.

## Synthesis after 20 experiments
- Best 0.6099 (exp 3) still stands; 17 discards in a row since. Eight variants of the model or feature set
  land in 0.6082-0.6091, i.e. the realistic level of the 11 basket features is ~0.609 and 0.6099 is at the top
  of the noise band (fold 4, ~400 customers, swings +-0.005).
- Dead ends (each tried once or twice): time of day/week, country (flag, TE), product identity (TE, text
  topics) -> seasonal drift kills these on fold 4; wholesale pack shares, ratios, relative price, variants,
  Christmas share -> redundant with quantity/price stats; depth 2/4, mcw 20, lr/trees, no subsampling -> flat.
- Remaining directions: (a) coarser histogram binning / monotone constraints (regularise how continuous
  features are split rather than how deep), (b) leaner feature set (exp 15 improved folds 1-3), (c) any
  genuinely new information about the customer type.

## Web research at experiment 20 (before 21)
- TDS next-purchase-day on Online Retail (https://towardsdatascience.com/using-machine-learning-to-predict-customers-next-purchase-day-7895ad49b4db):
  RFM + recency features with a 90-day label; for first-invoice-only data, recency/frequency are constant,
  confirming only monetary/basket features carry signal.
- Instacart XGBoost write-up (https://nbeguier.medium.com/machine-learning-2d4150bfd690): "preferable time of
  ordering" features - already tested as hour/weekday (exp 5), negative here.
- Takeaway: the literature has no further first-invoice signal; focus on regularising the split search.

### 21 - max_bin 256 -> 32 (model, regularise continuous splits)
Hypothesis: coarse quantile bins stop the trees from cutting continuous features (spend, prices) at
idiosyncratic thresholds, reducing variance across cohorts.
Result: discard, 0.6053 (0.604/0.591/0.599/0.628). Coarser bins lose resolution (fold 2 -0.009).

### 22 - monotone constraints on basket-size features (model, domain prior)
Hypothesis (UCI note: wholesalers return): churn probability should not rise with basket size. Constrain
n_lines, n_distinct_products, total_quantity, total_spend, max_quantity, max_line_total to be monotone
decreasing; price features unconstrained. Removes wiggly non-monotone fits that do not transfer.
Result: discard, 0.6072 (0.606/0.595/0.597/0.631). Slightly worse: the relationship is apparently not
monotone (e.g. very large single orders may be one-off events/bulk gifts).

### 23 - single, heavily smoothed product TE column (follow-up on 6, with research from Pargent et al.)
Hypothesis: exp 6 failed with four noisy TE columns at m=20. Per Pargent et al. (regularised TE), use only
the basket mean with strong smoothing (m=100), still 5-fold OOF on fit customers.
Result: discard, 0.6091 (0.612/0.607/0.601/0.617). FIRST REAL SIGNAL: folds 1-3 all clearly up vs best
(+0.005/+0.007/+0.001) and fold 4 down (-0.015). Diagnosis: fold 4 validates on the Sep-Nov 2010
pre-Christmas cohort; the Christmas products' TE is dominated by the Dec-2009 cohort, which is left-censored
(existing customers, churn 33%), so Christmas buyers in fold 4 look falsely loyal. TE is confounded by time.

### Web research after 3 consecutive discards (21-23, before 24)
- Towards AI, categorical encoding for time series (https://pub.towardsai.net/categorical-encoding-for-time-series-embracing-dynamic-and-meaningful-techniques-d5e796c07c8f):
  plain TE is biased under seasonality; adjust with a seasonal (month) term.
- "The When and How of Target Variable Transformations" (https://arxiv.org/pdf/2504.20821): remove the part of
  the target explained by time trends/seasonality before modelling.
- DataCamp (https://campus.datacamp.com/courses/intermediate-predictive-analytics-in-python/advanced-base-table-concepts?ex=1):
  seasonality distorts results if the timeline is built carelessly.

### 24 - product TE on the de-trended target (follow-up on 23)
Hypothesis: encode residual = churn - mean churn of the customer's first-invoice month (month means from fit
customers only), shrunk towards 0 (m=100). Products then carry their effect relative to their period, removing
the Dec-09 left-censoring and the upward churn trend from the encoding.
Result: discard, 0.6090 (0.611/0.604/0.601/0.621). Fold 4 recovered partly (0.617 -> 0.621), folds 1-3 still
above best. Direction right, but fold 4 still 0.011 under the best's lucky 0.632.

### 25 - de-trended product TE, m 100 -> 300 (follow-up on 24)
Hypothesis: most stock codes are bought by <50 fit customers; their TE is noise that hurts in a new season.
Stronger shrinkage keeps only well-supported product effects.
Result: KEEP, 0.6120 (0.613/0.603/0.606/0.627) vs 0.6099 (0.607/0.600/0.600/0.632). Folds 1-3 up, fold 4
within its noise. First gain in 22 experiments, and it comes from the product dimension the literature pointed
to (Kopczewska et al.), once the encoding is de-trended (Towards AI / arXiv 2504.20821) and heavily shrunk
(Pargent et al.). train.py now has a product_te() helper.

### 26 - TE shrinkage m 300 -> 1000 (follow-up on 25)
Hypothesis: if 100 -> 300 helped, the optimum may be higher still; tests the direction.
Result: KEEP, 0.6130 (0.615/0.609/0.605/0.623). Folds 1-2 up, 3 flat, 4 slightly down; +0.001 is within
noise, but the trend 100 -> 300 -> 1000 is monotone. Note: with m >> product count, te ~ sum(resid)/m, i.e.
the ranking approaches the product's SUMMED residual (popular products with consistent effects dominate).

### 27 - add basket sum of product TE (follow-up)
Hypothesis: the mean dilutes a few strong products in big baskets; the sum captures "how many loyal-type
products did they buy". Add prod_te_sum next to prod_te_mean.
Result: 0.6131 (0.613/0.608/0.608/0.623), +0.0001 over best. Logged as DISCARD on the program's simplicity
rule (gains < ~0.002 are noise; an extra feature for +0.0001 is not worth it). best_train.py/train.py reset to
the exp-26 version; a re-run reproduced 0.6130 exactly (verification only, not an experiment).

### 28 - TE on base stock code (follow-up)
Hypothesis: colour/size variants (79323P, 79323W) are the same product; encoding the numeric base code pools
their customers, giving more support per group and less noise.
Result: discard, 0.6098 (0.609/0.601/0.606/0.623). Folds 1-2 down: variants differ (e.g. colours sell to
different shops), so pooling loses signal.

### 29 - drop n_lines, now with TE (ablation, revisits 15)
Hypothesis: exp 15 raised folds 1-3 by dropping the near-duplicate n_lines; with the TE column in, the
simpler 11-feature set may now win overall.
Result: KEEP, 0.6161 (0.618/0.610/0.614/0.623) - every fold >= the previous best. Duplicate columns cost real
AUC via colsample dilution; ablation is a productive direction.

### 30 - drop total_quantity (ablation)
Hypothesis: total_quantity is largely implied by n_distinct_products x median/max quantity and total_spend.
Result: KEEP, 0.6185 (0.620/0.611/0.610/0.632). Folds 1, 2, 4 up; fold 3 down 0.0035.

## Synthesis after 30 experiments
- Best 0.6185 (from 0.5947 baseline): regularised XGBoost (+0.005), basket distribution stats (+0.010),
  de-trended + heavily shrunk product TE (+0.003), removing redundant columns n_lines/total_quantity (+0.0055).
- Key lessons: (1) product identity carries signal only once churn is de-trended by cohort month and shrunk
  hard (raw TE, text topics and pooled base codes all failed); (2) ADDING correlated columns hurt in 10+
  experiments, REMOVING them helps -> continue backward elimination; (3) model shape (depth, mcw, lr, bins,
  monotone) is flat.

## Web research at experiment 30 (before 31)
- First-mover bias in GB (https://arxiv.org/pdf/2603.22346): with correlated features, splits pick between
  near-identical signals on marginal differences; colsample interacts with this.
- Kuhn & Johnson FES, RFE chapter (https://bookdown.org/max/FES/recursive-feature-elimination.html): filter
  highly correlated features first; redundancy dilutes the ensemble.
- IFX feature starvation (https://cris.bgu.ac.il/en/publications/iterative-feature-exclusion-ifx-mitigating-feature-starvation-in-/):
  greedy boosting can over-rely on a few features.
Plan: continue one-at-a-time ablation of redundant columns (avg_unit_price, max_line_total, max_unit_price,
std_unit_price), then revisit colsample on the leaner set.

### 31 - drop avg_unit_price (ablation)
Hypothesis: mean unit price ~ total_spend / quantity and sits between min/max/std price; likely redundant.
Result: discard, 0.6144 (0.616/0.611/0.612/0.619). Fold 4 -0.014: average price level (cheap-goods vs.
premium shop) is a distinct signal. Keep it.

### 32 - drop max_line_total (ablation)
Hypothesis: max_line_total ~ max_quantity x price of that item; overlaps max_quantity and total_spend.
Result: discard, 0.6180 (0.618/0.616/0.612/0.627) - effectively a tie (-0.0005, folds 2-3 up, 4 down).
Strict rule (not higher) -> discard. max_line_total is close to redundant; candidate to drop later if a
leaner set needs it.

### 33 - drop std_unit_price (ablation)
Hypothesis: price spread is mostly max_unit_price - min_unit_price.
Result: discard, 0.6125 (0.615/0.612/0.605/0.618). All price stats carry distinct info; price-side ablation
is exhausted. The remaining 10 features look non-redundant.

### Web research after 3 consecutive discards (31-33, before 34)
- MetricGate target-encoding pitfalls (https://metricgate.com/blogs/pitfalls-of-target-encoding/) and docs
  (https://metricgate.com/docs/target-encoding-analysis/): OOF for training rows + global smoothed encoding for
  test rows; more folds -> lower-variance OOF estimates closer to the full-data encoding; 5 is the default.
- category_encoders wrapper (https://contrib.scikit-learn.org/category_encoders/_modules/category_encoders/wrapper.html):
  defaults to 5 folds.

### 34 - TE folds 5 -> 10 (follow-up)
Hypothesis: OOF values from 90% (vs 80%) of fit customers are closer in distribution to the full-fit
encoding used on validation, reducing the train/apply mismatch of prod_te_mean.
Result: discard, 0.6152 (0.616/0.610/0.606/0.629), all folds slightly down. 5 folds stays.

### 35 - de-trend TE by ISO week instead of month (follow-up)
Hypothesis: churn base rate shifts within months (e.g. pre-Christmas weeks); weekly cohort means remove more
of the seasonal confound from product effects. Risk: weekly means are noisier (~80 customers/week).
Result: KEEP, 0.6187 (0.620/0.610/0.614/0.632) vs 0.6185 (0.620/0.611/0.610/0.632). A tie (+0.0002, fold 3
+0.004, fold 2 -0.002) at zero complexity cost, so kept per the rule. Renamed month -> cohort afterwards
(no-op refactor, re-run reproduced 0.6187).

### 36 - colsample_bytree 0.8 -> 0.6 (model, follow-up on research at 30)
Hypothesis (first-mover-bias paper): lower column sampling spreads splits across correlated basket features
instead of letting one dominate; with 10 features each tree sees 6.
Result: KEEP, 0.6207 (0.621/0.615/0.614/0.634): all four folds up.

### 37 - colsample_bytree 0.6 -> 0.4 (follow-up)
Hypothesis: continue the direction; each tree sees 4 of 10 features -> stronger decorrelation of trees.
Result: discard, 0.6186 (0.623/0.613/0.612/0.626). Fold 4 -0.008: too few features per tree to form the
useful TE x basket interactions. 0.6 is the sweet spot.

### 38 - n_estimators 300 -> 500 (model)
Hypothesis: with colsample 0.6 each tree is weaker/more random; more trees may be needed to converge.
Result: discard, 0.6165 (0.616/0.612/0.610/0.629), all folds down -> 300 trees already past the optimum.

### 39 - n_estimators 300 -> 200 (model, opposite direction)
Hypothesis: if 500 overfits, fewer trees (implicit early stopping; harness gives no eval set) may help.
Result: KEEP, 0.6211 (0.621/0.614/0.613/0.636). +0.0004 = tie, but cheaper/simpler, so kept.

### 40 - n_estimators 200 -> 150 (model, follow-up)
Hypothesis: check whether the optimum is lower still (total shrinkage 150 x 0.03 = 4.5).
Result: discard, 0.6201 (0.620/0.613/0.612/0.635). Slightly under 200; optimum is ~200 trees.

## Synthesis after 40 experiments
- Best 0.6211 (exp 39). Gains since exp 30: TE de-trended by week (tie), colsample 0.6 (+0.002, all folds),
  200 trees (tie). Ablation is exhausted on the price side (avg/std price matter, max_line_total is a tie).
- Model settings now look tuned: depth 3, mcw 5, lr 0.03, 200 trees, subsample 0.8, colsample 0.6.
- The one information source that worked is label-based product effects, de-trended and shrunk. Next: extend
  the same recipe to other keys (description words, country), and re-test earlier-failed features under the new
  configuration only if there is a specific reason.

## Web research at experiment 40 (before 41)
- Pargent et al. benchmark via emergentmind (https://www.emergentmind.com/papers/2104.00629) and DataCamp Kaggle
  course (https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/feature-engineering-43ab8287-5e15-4fd2-930f-837293e73473?ex=7):
  regularised OOF TE is the best general-purpose encoder; smoothing pulls rare levels to the prior.
- scikit-learn TargetEncoder example (https://scikit-learn.org/1.6/auto_examples/preprocessing/plot_target_encoder.html):
  cross-fitted TE for high-cardinality features.
- No source on word-level TE; applying the same safeguards (OOF, shrinkage, de-trending) to description words.

### 41 - de-trended description-word TE (exploration, extends the TE recipe)
Hypothesis: words (LUNCH, BAG, CANDLE, ...) pool many stock codes, so their effects are better supported and
more stable across seasons than single stock codes. Add word_te_mean: basket mean over (customer, word) pairs
of each word's de-trended churn effect (same OOF folds and shrinkage as products).
Result: discard, 0.6179 (0.617/0.614/0.610/0.631). Folds 1, 3, 4 down. Words are too generic (RED, SET,
HEART appear everywhere): their effects average out to noise, and the column is largely redundant with
prod_te_mean.

### 42 - spend-weighted product TE (follow-up on TE)
Hypothesis: a basket's character is set by where the money goes; weight each product's TE by its line total
instead of a plain mean over distinct products (replaces prod_te_mean, no new column).
Result: discard, 0.6185 (0.619/0.612/0.612/0.632), all folds slightly down. Plain mean is better; spend
weighting lets one bulk line dominate.

### Web research after 3 consecutive discards (40-42, before 43)
- "Time Aggregation Features for XGBoost Models" (https://arxiv.org/html/2601.10019v1), read in full: their
  time-aware TE uses an expanding history (only rows strictly earlier than the encoded row), smoothing m=100
  towards a time-varying prior, and a history-volume feature log(1+count). No decay weighting.

### 43 - time-ordered (expanding) OOF for the fit-side product TE (follow-up on research)
Hypothesis: validation customers are encoded from strictly earlier customers. Random 5-fold OOF lets fit
customers be encoded from later customers too (e.g. next season's buyers of a product), a mismatch. Split
fit customers into 5 chunks by first-invoice time; chunk k is encoded from chunks < k (chunk 0 -> 0).
Result: discard, 0.6143 (0.603/0.606/0.616/0.633). Folds 1-2 (smallest fit sets) drop sharply: chunk 0 gets
no encoding and early chunks are encoded from little data, so the model learns a weaker TE relationship.
Random OOF uses 80% of fit for every customer, which matters more than the temporal mismatch here.

### 44 - add product popularity (history volume) feature (follow-up on research)
Hypothesis (arXiv 2601.10019 history-volume feature): mean log(1 + number of fit customers who bought the
product) over the basket. Tells the model how much support prod_te_mean has (niche vs mainstream basket),
and niche-product buyers may be specialist shops. Label-free.
Result: discard, 0.6183 (0.619/0.616/0.615/0.624). Folds 2-3 up, fold 4 down 0.012: popularity drifts by
construction (later customers buy new-season products with little fit history), the drift check would flag
it. Same failure pattern as the label-free product features in 6 and 11.

### 45 - TE shrinkage m 1000 -> 3000 (follow-up on 25/26)
Hypothesis: the trend 100 -> 300 -> 1000 was monotone; check whether it keeps going now that the model has
changed (colsample 0.6, 200 trees, 10 features).
Result: discard, 0.6188 (0.620/0.614/0.611/0.631), all folds slightly down. m=1000 is the optimum.

### Web research after 3 consecutive discards (43-45, before 46)
- Racklify, wholesale vs retail customers (https://racklify.com/encyclopedia/wholesale-customer-vs-retail-customer-key-differences-for-merchants/),
  Shopify B2B vs B2C (https://shopify.ca/enterprise/b2b-vs-b2c-ecommerce), Reveation classification criteria
  (https://www.reveation.io/blog/b2b-manufacturer-distributor-classification-criteria): resellers buy above
  normal quantities at lower unit prices; raw quantity alone is unreliable because norms differ per item.

### 46 - quantity relative to the product's typical quantity (exploration)
Hypothesis: buying 4x the usual quantity of an item is a reseller signal regardless of the item. Add
mean_rel_quantity = basket mean of log(quantity / median quantity of that stock_code over fit lines);
unknown products -> 0.
Result: discard, 0.6180 (0.618/0.613/0.614/0.627). Fold 3 up, others down. Every added basket column since
exp 3 has hurt; the 10-column set seems to be the sweet spot for this model.

### 47 - reg_lambda 1 -> 5 (model, from research at 10)
Hypothesis: L2 on leaf weights shrinks leaves supported by few customers (the TE column creates fine splits).
Result: KEEP, 0.6218 (0.622/0.614/0.618/0.634). Folds 1 and 3 up, 2 and 4 flat.

### 48 - reg_lambda 5 -> 20 (follow-up)
Hypothesis: test the direction; stronger L2 also slows effective learning (like lower lr).
Result: KEEP, 0.6220 (0.620/0.614/0.616/0.639): tie (+0.0002), fold 4 up, fold 1 down. Kept (zero cost).

### 49 - n_estimators 200 -> 300 under reg_lambda 20 (follow-up)
Hypothesis: lambda 20 shrinks each leaf a lot (hessian sums are small for logloss), so the ensemble may now
be under-fitted at 200 trees.
Result: discard, 0.6207 (0.620/0.615/0.615/0.634). Fold 4 -0.005. 200 trees stays.

### 50 - max_depth 3 -> 4 under reg_lambda 20 (model)
Hypothesis: depth 4 failed in exp 19 without L2; with lambda 20 the extra leaves are shrunk, so 3-way
interactions (TE x price x size) might now be learned without overfitting.
Result: discard, 0.6207 (0.617/0.613/0.615/0.638). Folds 1-2 down. Depth 3 confirmed again.

## Synthesis after 50 experiments
- Best 0.6220 (exp 48): 9 basket stats + de-trended product TE; XGB depth 3, 200 trees, lr 0.03, subsample
  0.8, colsample 0.6, mcw 5, lambda 20. Gains 40-50: reg_lambda (+0.0009).
- Since exp 30 every ADDED column has failed (words TE, popularity, rel quantity, ...), mostly via fold 4
  (latest cohort): label-free product-derived features drift; extra basket stats are redundant.
- Model is now well tuned (trees, depth, colsample, lambda all checked in both directions).
- Remaining budget 50. Plan: (a) buyer-profile features (product described by its buyers, label-free,
  not label-drifting); (b) swap rather than add columns (e.g. replace max_line_total, a tie in ablation);
  (c) small TE refinements (de-trend window, prior handling for unseen products).

## Web research at experiment 50 (before 51)
- EUR thesis on Tmall repeat buyers (https://thesis.eur.nl/pub/72505/Thesis-Draft-Ver3.0.pdf): item / brand /
  category "feature profiles" + XGBoost/SHAP; describing items by their buyers.
- IJCCC repeat-buyer feature engineering (https://univagora.ro/jour/index.php/ijccc/article/view/4988):
  user-merchant-item cross features.

### 51 - buyer-profile feature: products described by their buyers' spend (exploration)
Hypothesis: a product bought mostly by big-basket (reseller) customers marks the buyer as a reseller even if
their own basket is small (e.g. a first trial order). Feature buyer_spend_profile = basket mean over products
of mean log1p(total_spend) of the fit customers who bought that product; fit side computed 5-fold OOF so a
customer's own spend never enters; unseen products ignored (NaN if none known).
Result: discard, 0.6217 (0.618/0.616/0.615/0.638) - a tie with best (0.620/0.614/0.616/0.639). No gain.

### Web research after 3 consecutive discards (49-51, before 52)
- Train in Data, target encoder (https://blog.trainindata.com/target-encoder-a-powerful-categorical-encoding-method/):
  unknown levels -> prior by default, or ignore them -> missing.
- treeboost encoding (https://docs.rs/crate/treeboost/0.1.0/source/src/encoding/mod.rs): explicit "Unknown"
  bucket so the model learns how unseen levels behave.
- Larionov, Bayesian TE (https://arxiv.org/pdf/2006.01317v1): the prior point mean can mislead for unseen levels.
- Dataiku GB + categoricals (https://blog.dataiku.com/how-do-gradient-boosting-algorithms-handle-categorical-variables).

### 52 - product TE averaged over known products only (follow-up on research)
Hypothesis: unseen stock codes currently enter the basket mean as 0, so customers with many new-season
products (more common in later validation cohorts than in random-OOF fit rows) get a mean diluted towards 0.
Averaging over known products only removes this train/apply mismatch; baskets with no known product -> NaN.
Result: KEEP, 0.6223 (0.621/0.611/0.620/0.637). Fold 3 +0.004, fold 1 flat, folds 2/4 -0.002. Small but the
code is simpler (no fillna).

### 53 - ignore rare products (< 5 fit buyers) in the TE mean (follow-up)
Hypothesis: with m=1000, products with few buyers have TE ~0 and dilute the mean like unseen ones did.
Restrict the mean to products with >= 5 fit buyers.
Result: discard, 0.6202 (0.622/0.610/0.616/0.632). Folds 3-4 down: rare products carry information after all
(near-0 values pull the mean towards neutral appropriately, i.e. the shrinkage is doing its job).

### 54 - TE shrinkage m 1000 -> 500 under known-only averaging (follow-up)
Hypothesis: the optimal m was found while unseen products were 0-filled; with known-only averaging the
balance may shift towards less shrinkage.
Result: discard, 0.6214 (0.622/0.609/0.616/0.639). Fold 2 down. m=1000 stays.

### 55 - product TE on churn residualised for cohort AND basket spend (exploration)
Hypothesis: prod_te_mean partly re-encodes basket size (products that wholesalers buy look loyal), which the
trees already have. Residualising churn also on fit spend decile (additively: y - week mean - (spend-decile
mean - overall mean)) makes the TE capture product effects orthogonal to basket size: more complementary.
Result: discard, 0.6179 (0.621/0.614/0.609/0.628). Folds 3-4 down: the "wholesale product" part of the TE is
useful (products identify resellers better than spend does), removing it hurts.

### Web research after 3 consecutive discards (53-55, before 56)
- XGBoost docs (https://xgboost.readthedocs.io/en/release_1.0.0/parameter.html) and colsample explainer
  (https://medium.com/analytics-vidhya/xgboost-colsample-by-hyperparameters-explained-6c0bac1bdc1):
  bytree/bylevel/bynode multiply; bynode = random-forest-style per-split sampling.
- arXiv 2601.08121 (https://arxiv.org/html/2601.08121v1): intra-tree column subsampling hinders learning of
  ratio-like interactions (e.g. spend vs quantity) -> bynode may hurt; test once.

### 56 - colsample_bytree 0.6 -> colsample_bynode 0.6 (model)
Hypothesis: per-split sampling decorrelates more strongly than per-tree; research warns it can block
multi-feature interactions. Empirical check.
Result: discard, 0.6215 (0.620/0.614/0.615/0.637). Slightly below bytree, consistent with the research.

### 57 - subsample 0.8 -> 0.6 (model)
Hypothesis: stronger row bagging = more variance reduction for a noisy label; with lambda 20 the leaves
are already conservative, so fewer rows per tree may not hurt.
Result: KEEP, 0.6229 (0.618/0.614/0.617/0.642). Folds 4 +0.005, 2 +0.003; fold 1 -0.002.

### 58 - subsample 0.6 -> 0.5 (follow-up)
Hypothesis: continue the bagging direction.
Result: KEEP, 0.6231 (0.617/0.614/0.616/0.646). Tie (+0.0002) driven by fold 4 only (folds 1-3 slightly
down); kept per rule at zero cost, but I will not push subsample further on fold-4-only evidence.

### 59 - min_child_weight 5 -> 10 (model)
Hypothesis: mcw 20 failed in exp 10 with the old setup; with bagging 0.5, leaves see half the customers, so
a moderate increase may stabilise leaf estimates.
Result: discard, 0.6220 (0.616/0.612/0.614/0.645), folds 1-3 slightly down.

### 60 - min_child_weight 5 -> 2 (model, opposite direction)
Hypothesis: lambda 20 + bagging already regularise; smaller mcw allows splits isolating small high-signal
groups (e.g. large foreign wholesalers).
Result: KEEP, 0.6238 (0.617/0.614/0.616/0.648). Folds 2-4 up slightly vs best.

## Synthesis after 60 experiments
- Best 0.6238. Gains 50-60: known-only TE mean (+0.0003), subsample 0.8 -> 0.5 (+0.0008), mcw 5 -> 2
  (+0.0007). These are all small and partly fold-4-driven; the substantive structure has not changed since
  exp 36: 9 basket stats + de-trended shrunk product TE, colsample 0.6, strong L2.
- Failed 50-60: TE on rare-only/known-only variants, m 500, spend-residualised TE, bynode, mcw 10, buyer
  profile (tie). Lesson: the regularisation now comes from lambda + bagging rather than mcw.
- Next: lr/trees rebalance under heavy bagging; then retry the two near-tie features (buyer_spend_profile,
  prod_te_sum) under the new, more regularised model, since tolerance to extra columns may have changed.

## Web research at experiment 60 (before 61)
- scikit-learn GB regularisation example (https://scikit-learn.org/0.22/auto%5Fexamples/ensemble/plot_gradient_boosting_regularization.html):
  shrinkage + subsampling (stochastic GB) reduces variance; subsampling without shrinkage does poorly.
- Ridgeway, gbm vignette (https://anorien.csc.warwick.ac.uk/CRAN/web/packages/gbm/vignettes/gbm.pdf):
  learning rate and iterations trade off; use the smallest practical rate and pick iterations by CV.

### 61 - lr 0.03 -> 0.015 with 400 trees (model, follow-up on research)
Hypothesis: with subsample 0.5, smaller steps over more trees average over more bags -> less variance.
Result: discard, 0.6224 (0.620/0.613/0.615/0.642). Fold 1 up, fold 4 down. lr 0.03 x 200 stays.

### 62 - re-test buyer_spend_profile under the current model (follow-up on 51)
Hypothesis: exp 51 tied (0.6217 vs 0.6220). The model is now more strongly bagged (subsample 0.5, mcw 2),
which may make better use of an extra, partially redundant column.
Result: discard, 0.6205 (0.617/0.610/0.615/0.641). Worse than under the old model. Buyer profile is dead.

### 63 - drop max_line_total under current model (ablation, revisits 32)
Hypothesis: exp 32 was a near-tie (-0.0005). With heavier bagging, one fewer redundant column may now win,
and it is simpler.
Result: KEEP, 0.6249 (0.618/0.614/0.622/0.645). Fold 3 +0.006, others flat. 9 features now.

### 64 - drop median_quantity (ablation)
Hypothesis: median line quantity overlaps max_quantity and total_spend / n_distinct_products.
Result: discard, 0.6228 (0.619/0.613/0.620/0.640). Fold 4 -0.006: typical line size is distinct info.

### 65 - drop max_quantity (ablation)
Hypothesis: with median_quantity and total_spend present, the single largest line may be redundant.
Result: discard, 0.6238 (0.619/0.615/0.619/0.642). Near-tie (-0.001); keep max_quantity.

### 66 - drop max_unit_price (ablation)
Hypothesis: price spread is captured by std_unit_price and avg_unit_price; max is the most outlier-driven.
Result: KEEP, 0.6252 (0.621/0.615/0.622/0.644). Folds 1-2 up, 3 flat, 4 -0.002. Simpler, 8 features.

### 67 - drop min_unit_price (ablation)
Hypothesis: with avg and std of price present, the minimum (cheapest item, often a small accessory) may add
little.
Result: KEEP, 0.6260 (0.620/0.613/0.624/0.647). Folds 3-4 up, 1-2 slightly down. 7 features.

### 68 - drop std_unit_price (ablation)
Hypothesis: with max/min gone, std may be the remaining spread carrier (keep) or redundant with avg (drop).
Result: discard, 0.6186 (0.617/0.607/0.614/0.637), all folds down: std is the spread carrier. Keep.

### 69 - drop max_quantity in the 7-feature set (ablation, revisits 65)
Hypothesis: exp 65 was a near-tie with 9 features; with fewer columns the trade-off may differ.
Result: discard, 0.6226 (0.621/0.614/0.619/0.637). Fold 4 -0.010. Ablation is now exhausted: every one of the
7 remaining columns has been tested for removal (n_distinct_products/total_spend are the core size signals).

### 70 - colsample_bytree 0.6 -> 0.8 with 7 features (model re-tune)
Hypothesis: 0.6 was tuned with 10 features (6 per tree); with 7 features it gives only 4 per tree, possibly
too few for the TE x size x price interactions.
Result: discard, 0.6257 (0.622/0.612/0.623/0.645): tie. colsample 0.6 stays.

## Synthesis after 70 experiments
- Best 0.6260 (exp 67) with 7 features: n_distinct_products, total_spend, avg_unit_price, median_quantity,
  max_quantity, std_unit_price, prod_te_mean. XGB: 200 trees, depth 3, lr 0.03, subsample 0.5, colsample 0.6,
  mcw 2, lambda 20.
- 60-70: ablation under the heavier-regularised model paid off (+0.0022: drop max_line_total, max_unit_price,
  min_unit_price). All 7 remaining columns are confirmed necessary (each removal hurts).
- Next: explicitly engineered ratios (research below), then targeted re-tests of a couple of earlier ideas.

## Web research at experiment 70 (before 71)
- arXiv 2601.08121, read in full (https://arxiv.org/html/2601.08121v1): column subsampling hinders learning
  ratio-like interactions from raw components; including the engineered ratio removes the problem; authors
  recommend adding key ratios explicitly. (Exp 14 added three ratios at once under the old, less regularised
  model with redundant columns; one targeted ratio in the lean set is a different test.)

### 71 - add spend_per_product = total_spend / n_distinct_products (follow-up on research)
Hypothesis: average line value (order depth per product) is the canonical ratio of two kept features; trees
with colsample 0.6 often cannot see both parts together.
Result: discard, 0.6243 (0.621/0.615/0.621/0.641). Folds 1-2 up slightly, 3-4 down. Adding columns still
costs; the ratio info seems already available (median_quantity x avg_unit_price ~ line value).

### 72 - replace std_unit_price with price_cv = std / mean (swap, ratio)
Hypothesis: same column count; the coefficient of variation is a scale-free spread measure (a shop buying a
range of price points vs. a single price tier), which per the ratio research trees cannot form themselves.
Result: discard, 0.6238 (0.620/0.608/0.622/0.646). Fold 2 -0.005. The absolute spread is what matters.
Engineered ratios do not help here; the research's synthetic finding does not transfer.

### 73 - replace avg_unit_price with median_unit_price (swap)
Hypothesis: the mean price is pulled by one expensive item (e.g. a cabinet); the median is the typical price
point of the basket, a more robust "shop type" signal.
Result: discard, 0.6197 (0.620/0.611/0.616/0.633). Folds 3-4 down: the mean (incl. expensive items) is the
better signal. Swaps exhausted.

### Web research after 3 consecutive discards (71-73, before 74)
- Carey, hierarchical blending in TE (https://towardsdatascience.com/exploring-hierarchical-blending-in-target-encoding-fea4c59b305b),
  citing Micci-Barreca (2001): low-volume/unseen codes shrink towards the next level up of the code hierarchy
  instead of the global mean.
- Pargent et al. (https://link.springer.com/article/10.1007/s00180-022-01207-6): keep encodings OOF.

### 74 - hierarchical product TE: stock code -> 3-char code family -> 0 (follow-up on research)
Hypothesis: stock codes are allocated in series (22xxx, 84xxx, 85xxx ...), so the 3-char prefix is a crude
product family/collection. Family effects are better supported than single products; shrinking each product
to its family's TE (instead of 0) keeps information for rare products. Both levels from the same OOF split.
Result: discard, 0.6169 (0.612/0.605/0.607/0.644). Folds 1-3 down ~0.01. With m=1000 nearly every product
collapses to its family value, and code prefixes are not real product families (they mix unrelated items),
so the product-specific signal is lost. Like exp 28 (base codes): pooling products hurts.

### 75 - learning_rate 0.03 -> 0.05 with 120 trees (model, opposite of 61)
Hypothesis: 61 (smaller steps) was worse; check larger steps / fewer trees under heavy bagging.
Result: discard, 0.6228 (0.621/0.611/0.623/0.637). Fold 4 -0.010. lr 0.03 x 200 confirmed from both sides.

### 76 - add prod_te_min (most loyal-associated product in the basket) (follow-up on TE)
Hypothesis: a single strongly "returning-customer" product (e.g. a reseller staple) can mark a reseller even
in a mixed basket; the mean dilutes it. One extra column (exp 6 tested min/max only with raw, unshrunk TE).
Result: discard, 0.6209 (0.616/0.609/0.619/0.640), all folds down. The min is dominated by noise products.

### Web research after 3 consecutive discards (74-76, before 77)
- XGBoost tree methods (https://xgboost.readthedocs.io/en/release_1.6.0/treemethod.html): hist = histogram
  approximation; exact = all split candidates; older 'auto' used exact for small data.
- Dataiku GB flavours (https://blog.dataiku.com/the-many-flavors-of-gradient-boosting-algorithms): tree method
  barely changed accuracy on their benchmark.
- My exp 21: coarser bins (32) cost -0.005 -> resolution matters here; test the exact end.

### 77 - tree_method hist -> exact (model)
Hypothesis: exact split search on 2-3k customers uses all thresholds of total_spend / prices / TE.
Result: discard, 0.6253 (0.622/0.613/0.626/0.640): a tie (folds 1-3 up, 4 down), 2.5x slower. hist stays.

### 78 - gamma 0 -> 1 (model, from research at 10)
Hypothesis: min loss reduction per split prunes the weakest splits, which with lambda 20 are those on small
noisy groups; complements mcw 2.
Result: discard, 0.6256 (0.621/0.612/0.625/0.644): tie. Hyperparameters are saturated: exact, gamma, lr,
trees, depth, mcw, colsample and subsample have all been checked around the current point.

### 79 - product TE keyed on (stock_code, bulk flag) (exploration, TE refinement)
Hypothesis: buying product X in bulk (quantity >= 12) vs. as a single unit are different behaviours; a
reseller staple bought in bulk is a stronger return signal than the same item bought once. Key the TE on
stock_code + bulk flag (more levels, but m=1000 shrinkage handles sparsity).
Result: discard, 0.6199 (0.618/0.608/0.616/0.637), all folds down. Splitting products by quantity halves
their support; quantity is already a separate feature.

### Web research after 3 consecutive discards (77-79, before 80)
- Larionov, sampling-based Bayesian TE (https://arxiv.org/abs/2006.01317): generalisation suffers when
  training-time encodings differ from test-time encodings; their sampling targets that mismatch (PDF could not
  be rendered here, abstract only).
- This prompted an audit of my own train/apply mismatch: with m=1000 >> product counts, te ~ sum/m, so its
  SCALE is proportional to the amount of data used. OOF values (from 4/5 of fit customers) are ~0.8x the
  apply values (from all fit customers): every tree threshold on prod_te_mean is shifted at apply time.

### 80 - scale-consistent OOF TE: smoothing m*(K-1)/K for the fit side (follow-up, bug-like fix)
Hypothesis: using m' = 1000 * 4/5 = 800 on the OOF side makes fit and apply encodings the same scale
(sum*0.8/(count*0.8 + 800) = 0.8*sum/(0.8*(count+1000)) = sum/(count+1000)), removing the threshold shift.
Result: discard, 0.6217 (0.617/0.613/0.613/0.644). Folds 1 and 3 down. Counter-intuitive: the scale shift
seems to act as extra shrinkage of the TE at apply time relative to what the trees learned, which helps a
drifting feature. Left as is (documented as a known, apparently benign, mismatch).

## Synthesis after 80 experiments
- Best 0.6260 (exp 67), unchanged for 13 experiments. 70-80 all discarded: ratios (2), swaps (2), hierarchical
  TE, lr/trees, TE min, exact, gamma, bulk-keyed TE, scale-consistent TE. Three were ties (0.6253-0.6257).
- The model + 7-feature set is at a local optimum; most perturbations land within -0.001..-0.006.
- With 20 experiments left: (1) re-test the strongest EDA signal not yet tested under the regularised model
  (hour of day: EDA showed evening buyers churn much less); (2) TE de-trend window; (3) final clean-up.

## Web research at experiment 80 (before 81)
- Optimove, first-timer churn (https://www.optimove.com/?p=25845): first- and second-order attributes,
  including time of day, are highly correlated (vendor claim, no effect size).
- Rewix, B2B next-order prediction (https://www.rewixecommerce.com/en/magazine/predicting-the-next-order-in-b2b-ecommerce)
  and TopGrowth 90-day retention guide (https://topgrowthmarketing.com/ecommerce-customer-retention-90-day-guide/):
  B2B reorder cadence vs consumer one-offs.

### 81 - re-test hour of day (alone) under the regularised model (follow-up on 5 + research)
Hypothesis: exp 5 (hour + weekday, old depth-3/lambda-1 model) lost 0.013, probably by overfitting noise
splits. EDA: evening first orders (18-20h) churn 9-40% vs ~45%. With lambda 20, subsample 0.5 and only the
fractional hour (no weekday), the model may now use the real part of the signal.
Result: discard, 0.6173 (0.615/0.606/0.616/0.632), all folds down ~0.008. Time of day is definitively
harmful here (the evening effect rests on ~100 customers and does not transfer across cohorts).

### 82 - de-trend product TE by 14-day cohorts instead of ISO weeks (TE refinement)
Hypothesis: weekly cohort means (~60-80 customers) are noisy; 14-day windows halve that noise while still
following the seasonal trend more closely than months (exp 35 found week >= month by a hair).
Result: discard, 0.6247 (0.620/0.611/0.621/0.647): tie (-0.0013). Weekly stays.

### Web research after 3 consecutive discards (80-82, before 83)
- XGBoost "Random Forests in XGBoost" tutorial (https://xgboost.readthedocs.io/en/latest/tutorials/rf.html):
  num_parallel_tree > 1 grows a forest per boosting round (boosted random forest), needs subsample < 1 and
  column sampling < 1 for randomness - both already set.
- Packt, XGBoost random forests (https://subscription.packtpub.com/book/data/9781839218354/11/ch11lvl1sec56/finding-xgboost-random-forests).

### 83 - num_parallel_tree 1 -> 4 (model, follow-up on research)
Hypothesis: with subsample 0.5, each round's single tree is a noisy draw; averaging 4 bagged trees per round
reduces that variance (boosted random forest).
Result: discard, 0.6255 (0.622/0.615/0.623/0.642): tie. Folds 1-2 up, fold 4 down; 2.5x slower. Not kept.

### 84 - replace max_quantity with q90_quantity (swap, robustness)
Hypothesis: the single largest line is an outlier-prone statistic (one 1,000-unit line); the 90th percentile
of line quantities describes the bulk part of the basket more stably.
Result: discard, 0.6213 (0.619/0.611/0.615/0.639). Folds 3-4 down: the extreme line itself (a single big
bulk order) is the informative part.

### 85 - TE folds 5 -> 3 (follow-up on 34 and 80)
Hypothesis: exp 34 (10 folds) and exp 80 (scale-consistent OOF) both hurt, which suggests that fit-side TE
values being smaller/noisier than apply-side values acts as useful regularisation. 3 folds strengthen that
(OOF from 2/3 of the data). Tests the direction.
Result: discard, 0.6244 (0.622/0.612/0.617/0.647). Tie-ish; 5 folds is the optimum from both sides.

### Web research after 3 consecutive discards (83-85, before 86)
- XGBoost categorical tutorial (https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) and NVIDIA
  blog (https://developer.nvidia.com/blog/categorical-features-in-xgboost-without-manual-encoding): native
  categorical splits via enable_categorical=True; optimal partitioning of category sets; max_cat_threshold
  regularises against overfitting.

### 86 - country as a native XGBoost categorical (exploration, third country representation)
Hypothesis: exps 4 (UK flag) and 16 (TE) failed; a partition split can group e.g. {Germany, France, Channel
Islands, ...} vs rest as the trees see fit, and only where it combines with basket features. Categories
fixed from fit lines (unseen countries -> NaN). Requires enable_categorical=True.
Result: discard, 0.6228 (0.618/0.610/0.619/0.645). Third country representation, third failure: country is
not useful for ranking within these validation windows (too few non-UK customers per window).

### 87 - reg_lambda 20 -> 50 (model, follow-up on 47/48)
Hypothesis: 1 -> 5 -> 20 were all >= previous; check whether the L2 trend continues on the 7-feature model.
Result: discard, 0.6242 (0.618/0.614/0.622/0.643). Folds 1 and 4 down; lambda 20 is the optimum.

### Web research after 3 consecutive discards (85-87, before 88)
- APXML regularised objectives (https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-3-regularization-gradient-boosting/regularized-objective-functions):
  L1 (alpha) can zero out leaf outputs (sparsity); L2 shrinks without zeroing.
- TDS visual guide (https://towardsdatascience.com/visually-understand-xgboost-lightgbm-and-catboost-regularization-parameters-aa12abcd4c17)
  and theneuralbase guide (https://theneuralbase.com/xgboost/learn/beginner/reg-alpha-and-reg-lambda-l1-l2/):
  tune lambda first; alpha helps when many leaves/features carry no signal.

### 88 - reg_alpha 0 -> 1 (model, follow-up on research)
Hypothesis: many depth-3 leaves on this weak signal have near-zero true effect; L1 zeroes them out instead
of letting each contribute a little noise.
Result: discard, 0.6246 (0.619/0.613/0.622/0.645). Tie-ish, slightly worse. L1 not needed.

### 89 - recency-weighted product TE (half-life 120 days) (exploration, drift; research at 17 and 43)
Hypothesis: the Google instance-decay work (exp 17 research) argues recent data is more relevant under
drift; model sample weights are not reachable through the harness, but the TE statistics are mine. Weight
each fit customer's residual by 0.5 ** (age_days / 120), age measured from the latest fit date, in both sum
and count. Unlike exp 43 (expanding chunks), every fit customer still gets a full-data OOF encoding.
Result: 0.6261 (0.619/0.614/0.619/0.653) vs best 0.6260 (0.620/0.613/0.624/0.647). +0.0001 = noise, with
fold 4 +0.006 and fold 3 -0.005; adds 4 lines. Logged as DISCARD under the simplicity rule (same precedent
as exp 27); best_train.py/train.py restored to the exp-67 version, re-run verified 0.6260 (not an experiment).
The fold-4 gain is consistent with the drift story, so one milder setting is worth a single follow-up.

### 90 - recency-weighted product TE, half-life 240 days (follow-up on 89)
Hypothesis: 120 days discards too much of the older data for the middle folds (fold 3 down); a milder decay
may keep fold 4's gain without hurting fold 3. Keep only if clearly above noise.
Result: 0.6262 (0.622/0.615/0.621/0.647), +0.0002: again a tie. Two half-lives both tie -> recency
weighting is neutral here. DISCARD under the simplicity rule; files restored to exp-67, re-run 0.6260.

## Synthesis after 90 experiments
- Best 0.6260 (exp 67), unchanged for 23 experiments. 80-90: hour (-0.009), country categorical (-0.003),
  14-day de-trend, TE folds 3, lambda 50, alpha 1, q90 quantity, parallel trees, recency-weighted TE (2 ties).
- The search has converged: every axis has been probed on both sides of the current point and lands within
  noise (+-0.002) or below. The final 10 experiments: last structural model variants (lossguide), then
  stop changing things that only chase fold-4 noise.

## Web research at experiment 90 (before 91)
- XGBoost tree methods (https://xgboost.readthedocs.io/en/stable/treemethod.html) and Packt grow_policy chapter
  (https://subscription.packtpub.com/book/data/9781800564480/6/ch06lvl1sec34): lossguide grows the leaf with
  the highest loss reduction (LightGBM-style), bounded by max_leaves; can make unbalanced trees.

### 91 - grow_policy lossguide, max_leaves 6, no depth limit (model, follow-up on research)
Hypothesis: leaf-wise growth spends splits where the signal is (e.g. deep in the large-basket region) and
stays shallow elsewhere, which depthwise depth-3 cannot.
Result: discard, 0.6252 (0.621/0.615/0.621/0.643). Tie (-0.0008). Depthwise stays.

### 92 - max_depth 3 -> 2 under the current regularisation (model)
Hypothesis: depth 2 was a tie in exp 8 on the old setup; with only 7 features and strong bagging, pairwise
interactions may suffice and transfer better.
Result: discard, 0.6243 (0.620/0.612/0.620/0.646). Depth 3 confirmed a third time.

### Web research after 3 consecutive discards (90-92, before 93)
- scikit-learn TargetEncoder (https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.TargetEncoder.html):
  smooth="auto" = empirical Bayes estimate of the shrinkage.
- MetricGate empirical Bayes shrinkage (https://metricgate.com/docs/empirical-bayes-shrinkage/): shrinkage factor
  = within-group sampling variance / (between-group variance + sampling variance); between-group variance by
  method of moments -> m ~ sigma2_within / tau2_between.

### 93 - data-driven (empirical Bayes) TE smoothing per fold (follow-up on research)
Hypothesis: m=1000 was hand-tuned on the CV mean; the right shrinkage depends on how much real
between-product variation each fold's fit data shows. Estimate m = var(resid) / tau2 with tau2 by method of
moments over product means (count-weighted), clipped to [100, 5000].
Result: discard, 0.6224 (0.618/0.611/0.615/0.645), all folds down. The in-sample variance decomposition
overstates real between-product variance (part of it is seasonal and does not transfer), so EB picks a
smaller m than the out-of-time optimum. The CV-tuned m=1000 is better: what matters is transfer to the
next season, which EB cannot see.

### 94 - colsample_bytree 0.6 -> 0.5 (model, fine-tune)
Hypothesis: 0.4 (-0.002) and 0.8 (tie) bracket 0.6; with 7 features 0.5 and 0.6 both give 3-4 columns per
tree, so check the finer step.
Result: discard, 0.6239 (0.619/0.613/0.622/0.642). 0.6 stays.

### Web research after 3 consecutive discards (92-94, before 95)
- Friedman's stochastic GB as summarised in Wikipedia (https://en.wikipedia.com/wiki/Boosted_trees): f ~ 0.5
  works well for small/moderate training sets; smaller f regularises but can add bias.
- APXML GBM subsampling (https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/gbm-subsampling-stochastic):
  too small a fraction hinders learning / needs more trees.

### 95 - subsample 0.5 -> 0.4 (model, bracket)
Hypothesis: the research says 0.5 is typical; 0.8 -> 0.6 -> 0.5 each helped slightly, so bracket below once.
Result: discard, 0.6243 (0.620/0.612/0.623/0.643). Fold 4 down; 0.5 confirmed, as the research predicts.

### 96 - min_child_weight 2 -> 1 (model, bracket)
Hypothesis: 5 -> 2 helped (+0.0007); bracket the next step down, relying on lambda 20 for regularisation.
Result: discard, 0.6256 (0.620/0.614/0.623/0.647): tie. mcw 2 stays.

### Web research after 3 consecutive discards (94-96, before 97)
- H2O target encoding (https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/target-encoding.html):
  sigmoid blending (inflection_point, smoothing) instead of linear count shrinkage; a different functional
  form, not pursued with only 4 experiments left (would need its own tuning).

### 97 - n_estimators 200 -> 250 (model, bracket under current regularisation)
Hypothesis: the tree count was last tuned (exp 39/40) before lambda 20, subsample 0.5 and mcw 2; those slow
learning, so the optimum may have shifted up.
Result: discard, 0.6260 (0.622/0.613/0.624/0.645): exactly equal to best -> discard per rule (equal is not
higher); 200 trees is simpler.

### 98 - TE smoothing m 1000 -> 1500 (TE fine-tune under the final model)
Hypothesis: m was bracketed (500, 3000) on earlier model states; check a finer step on the final 7-feature
model.
Result: discard, 0.6246 (0.621/0.612/0.620/0.646). Fold 3 down.

### 99 - TE smoothing m 1000 -> 700 (TE fine-tune, other side)
Hypothesis: complete the bracket around m=1000 on the final model.
Result: discard, 0.6252 (0.620/0.612/0.621/0.648). m=1000 is the optimum from both sides.

### 100 - de-trend TE by month instead of week, on the final model (robustness re-check of 35)
Hypothesis: week beat month by only +0.0002 on an earlier model state (exp 35); re-check on the final model
so the kept choice is not an artefact of that state.
Result: discard, 0.6234 (0.619/0.611/0.622/0.641). Week is clearly better on the final model (+0.0026).

## Final summary (100 experiments)
- Best CV AUC 0.6260 (exp 67), folds 0.620/0.613/0.624/0.647, vs baseline 0.5947: +0.031.
  19 keeps (baseline included), 81 discards, 0 crashes. train.py == best_train.py (exp-67 state, re-verified 0.6260).
- Final pipeline: 6 basket statistics (n_distinct_products, total_spend, avg_unit_price, median_quantity,
  max_quantity, std_unit_price) + prod_te_mean (basket mean of each stock code's churn effect, de-trended by
  first-invoice week, shrunk with m=1000, 5-fold OOF on fit, unseen codes ignored). XGBoost: 200 trees, depth 3,
  lr 0.03, subsample 0.5, colsample_bytree 0.6, min_child_weight 2, reg_lambda 20.
- Where the gain came from: model regularisation (+0.005), basket distribution stats (+0.010), de-trended
  shrunk product TE (+0.003), dropping redundant columns (+0.008 across exps 29/30/63/66/67), colsample /
  lambda / subsample / mcw (+0.005).
- What never worked: time of day/week, country (3 encodings), label-free product features (topics,
  popularity, buyer profiles - they drift across seasons), pooled/hierarchical product codes, ratios, extra TE
  aggregates, time-ordered or recency-weighted TE, empirical-Bayes smoothing.
- Caveat for the human: the signal is weak and the fold-4 window (~400 customers) swings +-0.005; many late
  keeps were +0.0002-0.0008. The held-out test result should be expected around 0.61-0.62, not exactly 0.626.
