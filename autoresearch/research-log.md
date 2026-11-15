# Research log: autoresearch run1

Metric: mean AUC-ROC over 4 time-ordered expanding-window folds (`prepare.run_cv`).
Noise floor: fold AUCs vary ~0.01, so gains < ~0.002 are treated as likely noise (but the keep rule in program.md is strictly "higher than best").

### Exp 1: baseline
- Type: baseline
- Hypothesis: Unmodified train.py, records the starting point.
- Result: CV AUC 0.5712 (best before 0.0000), folds [0.5668, 0.5769, 0.5794, 0.5616], 2.2s -> **keep**

## Research before exp 2
- Shallow trees (depth 3-6, ensembles fix errors), lr 0.01-0.05 with more trees, and L1/L2 penalties help on noisy data: [metricgate XGBoost tuning guide](https://metricgate.com/blogs/xgboost-hyperparameter-tuning-guide/).
- Tuning only a few params is often worse; pick the number of boosting rounds by early stopping: [arXiv 2602.05786](https://arxiv.org/pdf/2602.05786).
- Repeat-purchase/churn on the Online Retail data reaches roughly AUC 0.62 with RFM-T features on a time split, so the signal is weak: [churn modelling write-up](https://daffadaniel.github.io/05-ChurnModelling/), [TDS next purchase day](https://towardsdatascience.com/using-machine-learning-to-predict-customers-next-purchase-day-7895ad49b4db).
- Plan: (1) regularise hard (shallow, slow, subsampled); (2) prune noisy features (93 PCA columns); (3) add ratio/log features; (4) try linear and ensemble models.

### Exp 2: shallow, slow, subsampled XGB (regularisation)
- Type: exploration
- Hypothesis: Weak signal + 173 features: baseline depth-6 overfits; shallow/slow/subsampled trees should generalise better (metricgate guide).
- Result: CV AUC 0.5974 (best before 0.5712), folds [0.5831, 0.5891, 0.6124, 0.6048], 2.0s -> **keep**

_Refactor (not an experiment): moved the exp-2 hyperparameters into an `XGB_PARAMS` dict. Re-ran to confirm the result is identical (0.5974), then copied to best_train.py._

### Exp 3: max_depth 3->2, n_estimators 400->600
- Type: follow-up
- Hypothesis: Exp 2 showed regularisation is the main lever; even shallower trees (depth 2, mostly pairwise interactions) may generalise better.
- Result: CV AUC 0.5974 (best before 0.5974), folds [0.5798, 0.5959, 0.6097, 0.6043], 2.0s -> **discard**

### Exp 4: learning_rate 0.03->0.015, n_estimators 400->800
- Type: follow-up
- Hypothesis: Slower learning with more trees usually generalises a bit better (metricgate guide).
- Result: CV AUC 0.5939 (best before 0.5974), folds [0.5774, 0.5941, 0.6133, 0.5907], 3.6s -> **discard**

### Exp 5: colsample_bytree 0.5->0.3
- Type: exploration
- Hypothesis: Many redundant/noisy columns (93 PCA + 5 stats per aggregate); stronger column subsampling decorrelates trees.
- Result: CV AUC 0.5933 (best before 0.5974), folds [0.5809, 0.5905, 0.6081, 0.5937], 1.7s -> **discard**

## Research after 3 consecutive discards (exps 3-5)
- Leaving lambda=1 and gamma=0 is often suboptimal on noisy data; tune max_depth/min_child_weight first, then gamma/reg_lambda: [metricgate guide](https://metricgate.com/blogs/xgboost-hyperparameter-tuning-guide/), [Capital One XGBoost tips](https://www.capitalone.com/tech/machine-learning/how-to-control-your-xgboost-model/).
- Feature selection alone often does not fix XGBoost overfitting: [arXiv 2411.05937](https://arxiv.org/pdf/2411.05937). Feature pruning is still worth a test, but explicit regularisation comes first.
- Next: min_child_weight up, then reg_lambda/gamma.

### Exp 6: min_child_weight 10->30
- Type: exploration
- Hypothesis: Larger min_child_weight stops splits on small noisy leaves (Capital One / metricgate guides).
- Result: CV AUC 0.5987 (best before 0.5974), folds [0.5845, 0.5947, 0.6113, 0.6042], 1.8s -> **keep**

### Exp 7: min_child_weight 30->60
- Type: follow-up
- Hypothesis: Exp 6 helped a little; check whether more leaf-size regularisation keeps helping.
- Result: CV AUC 0.6023 (best before 0.5987), folds [0.5908, 0.5975, 0.6193, 0.6017], 1.7s -> **keep**

### Exp 8: reg_lambda 1->10 (ran with min_child_weight=30 by mistake, not best 60)
- Type: exploration
- Hypothesis: A stronger L2 penalty on leaf weights shrinks noisy leaf values (metricgate).
- Result: CV AUC 0.6010 (best before 0.6023), folds [0.5884, 0.5963, 0.6165, 0.6029], 1.9s -> **discard**

### Exp 9: gamma 0->1.0 (ran with min_child_weight=30 by mistake, not best 60)
- Type: exploration
- Hypothesis: A minimum loss reduction per split prunes weak splits (Capital One guide).
- Result: CV AUC 0.5987 (best before 0.6023), folds [0.5845, 0.5947, 0.6113, 0.6042], 1.8s -> **discard**

### Exp 10: reg_lambda 1->10 (on the true best, min_child_weight=60)
- Type: follow-up (re-test of exp 8)
- Hypothesis: Exp 8 accidentally ran on mcw=30; re-test L2 on the real best version.
- Result: CV AUC 0.6024 (best before 0.6023), folds [0.5885, 0.6008, 0.6159, 0.6044], 1.8s -> **keep**
- Note: +0.0001 is noise, but program.md's rule is strictly "higher -> keep", so it stays.

_Process note: after exp 10 the user asked for no heredocs or inline python. From exp 11 on, logs are written with the Edit tool, train.py is edited with Edit, and `autoresearch/run_experiment.py` only runs the CV and handles the keep/discard file copies._

## Synthesis after 10 experiments
- **Works:** regularisation. Shallow trees, lr 0.03 and subsampling took 0.571 to 0.597. Large min_child_weight (60) added about +0.005. reg_lambda=10 is neutral to slightly positive.
- **Doesn't work:** going shallower (depth 2), slower lr, or colsample 0.3. All flat or worse.
- Pure hyperparameter tweaks are reaching the noise floor (~0.002). The next block focuses on features: dropping or compressing noisy blocks (93 PCA columns, redundant min/max/mean/median/sum stats), the time trend (year/month), and ratio features.

## Research at exp 10
- First-to-second-purchase churn models usually score AUC 0.6-0.75. XGBoost on basic first-basket features plus product-category dummies works best, and the signal is weak: [encyclopedia.pub entry](https://encyclopedia.pub/entry/download_pdf/65c6f05e0e7d812ae7426084884fd930), [SOLUS first-to-repeat models](https://solus.ai/predictive-modelling-for-customer-retention-building-first-to-repeat-and-churn-models/).
- Seasonal effects cause drops ([CEU capstone](https://www.etd.ceu.edu/2019/pasztor_aurel.pdf)). On time-ordered folds, `year`/`month` can encode trend rather than stable signal, so test dropping them.
- Stacking XGBoost with another learner improved repeat-buyer AUC: [Erasmus thesis](https://thesis.eur.nl/pub/72505/Thesis-Draft-Ver3.0.pdf). Try a blend with logistic regression later.

### Exp 11: drop all pca_* columns
- Type: exploration
- Hypothesis: The 93 description-embedding PCA columns are mostly noise for a weak-signal target; dropping them may cut variance.
- Result: 0.6013 (best 0.6024), folds [0.5941, 0.5897, 0.6074, 0.6141] -> **discard**. Mixed: folds 1 and 4 up, 2 and 3 down. The PCA columns carry some signal (in line with the "product category dummies help" finding). Try keeping only the leading components later.

### Exp 12: drop `year`
- Type: exploration
- Hypothesis: On expanding-window time folds, validation rows sit in later years than most training rows, so `year` encodes trend that does not carry forward ([CEU capstone](https://www.etd.ceu.edu/2019/pasztor_aurel.pdf) on seasonal/time effects).
- Result: 0.6035 (best 0.6024), folds [0.5903, 0.5967, 0.6204, 0.6067] -> **keep** (small gain, 3 of 4 folds up vs exp 10).

### Exp 13: also drop `month`
- Type: follow-up
- Hypothesis: Same trend argument as exp 12, applied to month.
- Result: 0.6018, folds [0.5921, 0.5948, 0.6121, 0.608] -> **discard**. Month carries real seasonal signal (e.g. Q4 gift buyers behave differently).

### Exp 14: basket ratio features
- Type: exploration
- Hypothesis: Ratios such as spend per line, units per line and distinct products per line describe basket "shape" (wholesale vs. gift buyer), which trees can't easily build from raw totals with depth 3.
- Result: 0.6029, folds [0.5938, 0.5972, 0.6146, 0.6061] -> **discard** (within noise; avg_unit_price already captures most of it).

### Exp 15: keep only pca_00..pca_19
- Type: follow-up to exp 11
- Hypothesis: Exp 11 showed the PCA block carries some signal but also noise. The leading components hold most of the variance, while the tail adds splits on noise.
- Result: 0.5941, folds [0.5853, 0.592, 0.6029, 0.5959] -> **discard**. All folds worse, and worse than dropping all PCA (exp 11). The tail components matter: a few specific product directions may be predictive, not the leading variance.

## Research after 3 consecutive discards (exps 13-15)
- XGBoost column sampling is cumulative (bytree x bylevel x bynode). Per-split sampling (`colsample_bynode`) adds randomness at each split, which acts as a different kind of regulariser than per-tree sampling: [XGBoost parameter docs](https://xgboost.readthedocs.io/en/release_1.6.0/parameter.html), [colsample_by* explained](https://medium.com/analytics-vidhya/xgboost-colsample-by-hyperparameters-explained-6c0bac1bdc1).
- No direct evidence on LR blending came up. The Erasmus thesis (exp-10 research) supports stacking in general.
- Next: colsample_bynode, then an XGB + logistic-regression blend.

### Exp 16: colsample_bynode=0.5
- Type: exploration
- Hypothesis: Sampling at each split, rather than only per tree, gives random-forest-like decorrelation, which suits the noisy wide feature set (XGBoost docs).
- Result: 0.6037 (best 0.6035), folds [0.5878, 0.5999, 0.6075, 0.6195] -> **keep** by the strict rule. +0.0002 is noise; the fold pattern shifted (fold 4 up, fold 3 down).

### Exp 17: soft-vote blend of XGB + logistic regression
- Type: exploration
- Hypothesis: Stacking/blending different learners improved repeat-buyer AUC ([Erasmus thesis](https://thesis.eur.nl/pub/72505/Thesis-Draft-Ver3.0.pdf)). A heavily regularised LR on quantile-normalised features captures smooth monotone effects that shallow trees approximate poorly, and its errors should be weakly correlated with XGB's. The quantile transform is fitted inside the pipeline on training rows only.
- Result: **crash**: `ValueError: Input X contains NaN` in LogisticRegression. Some features (probably prior_* stats for customers without history) contain NaN, which XGB handles natively.

### Exp 18: same blend, with a median imputer + missing indicators in the LR branch
- Type: follow-up (fix of exp 17)
- Hypothesis: Same as exp 17. SimpleImputer(median, add_indicator=True) is fitted on training rows inside the pipeline, so the NaN pattern ("no prior history") stays available to the LR as indicator columns.
- Result: **0.6141** (best 0.6037), folds [0.5998, 0.6063, 0.6213, 0.6291] -> **keep**. All 4 folds up, about +0.010: the largest gain since exp 2. The linear model adds real, complementary signal.

### Exp 19: LR C 0.01 -> 0.003
- Type: follow-up
- Hypothesis: With ~180 features and weak signal, an even stronger L2 penalty may help the LR branch.
- Result: 0.6154 (best 0.6141), folds [0.6015, 0.6087, 0.6219, 0.6296] -> **keep**. All folds up slightly.

### Exp 20: LR C 0.003 -> 0.001
- Type: follow-up
- Hypothesis: Keep following the regularisation gradient for the LR branch.
- Result: **no-op**. The edit to train.py did not apply (whitespace mismatch), so the unchanged best (C=0.003) was re-run: 0.6154 with identical fold AUCs. It counts toward the 100 and is logged as discard. It confirms the pipeline is deterministic. C=0.001 is retested in exp 21.

## Synthesis after 20 experiments
- Best: **0.6154** = shallow regularised XGB (depth 3, lr 0.03, 400 trees, mcw 60, lambda 10, colsample 0.5 tree + 0.5 node), without `year`, soft-blended 50/50 with a strongly L2-regularised logistic regression (median imputer + missing indicators, quantile-normal transform, C=0.003).
- **Biggest levers:** (1) regularising XGB, +0.026; (2) blending in a linear model, +0.010 on every fold.
- **No help:** depth 2, a slower lr, colsample 0.3, dropping or truncating PCA, dropping month, basket ratios.
- **Next:** tune the blend (weights, LR penalty, L1 vs L2), try LR-only to see how much each branch contributes, try a third diverse learner (ExtraTrees/RandomForest), and revisit feature engineering for the LR branch.

## Research at exp 20
- Blends pay off through low prediction correlation between members. A weak member can drag a blend down unless weighted ([NVIDIA Kaggle Grandmasters playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data), [letsdatascience stacking/blending](https://letsdatascience.com/blog/stacking-and-blending-the-secret-weapons-behind-winning-competitions)).
- Rank averaging is calibration-invariant ([Kaggle S6E2 write-up](https://heart-s6e2.pages.dev/)). This matters here: a heavily regularised LR outputs probabilities squeezed near the base rate, so in a probability average its *effective* weight shrinks as C drops. Changing C changes both the LR fit and the blend weight. Test a rank-average blend.
- Greedy/weighted blending tuned on the target metric ([Weighted Ensemble L2](https://csvfit.pages.dev/blog/weighted-ensemble-l2)): explicit blend weights are worth a test.

### Exp 21: LR C 0.003 -> 0.001 (redo of exp 20's intent)
- Type: follow-up
- Hypothesis: Stronger L2 on the LR branch. Note the confound: smaller C also shrinks LR's effective blend weight.
- Result: 0.6152, folds [0.6022, 0.6086, 0.6205, 0.6295] -> **discard** (flat; C has plateaued).

### Exp 22: scale-free blend (z-scored logits) instead of a probability average
- Type: exploration
- Hypothesis: With probability averaging, the compressed LR outputs get less weight than XGB's. Standardising each member's logit with mean/std from its own *training* predictions (no validation information) and summing gives an honest 50/50 blend, like rank averaging but with state learned only from X_fit ([Kaggle S6E2 rank-blend write-up](https://heart-s6e2.pages.dev/)).
- Result: 0.6157 (best 0.6154), folds [0.6017, 0.6087, 0.622, 0.6302] -> **keep** by the strict rule (+0.0003 is noise). The useful part is that blend weights are now explicit and scale-free, so they can be tuned.

### Exp 23: blend weights XGB:LR 1:1 -> 1:2
- Type: follow-up
- Hypothesis: If LR adds most of the complementary signal, giving it more weight may help.
- Result: 0.6161 (best 0.6157), folds [0.6032, 0.6082, 0.6233, 0.6297] -> **keep** (noise-level).

### Exp 24: ablation, LR only (weights 0:1)
- Type: ablation
- Hypothesis: Measure how much XGB still contributes. If LR alone is about as good, the simpler model wins.
- Result: 0.6138, folds [0.6034, 0.6047, 0.6229, 0.6242] -> **discard**. Key insight: a regularised linear model on quantile-normalised features (0.6138) **beats the tuned XGB alone (0.6037)**. The signal is mostly smooth and monotone, and XGB adds about +0.002 in the blend. Next, focus on the linear branch: smooth nonlinearity via splines (GAM-like) and penalty type.

### Exp 25: LR branch gets SplineTransformer features (GAM-like)
- Type: exploration
- Hypothesis: If effects are smooth but non-monotone (e.g. spend has a sweet spot), per-feature cubic splines on the quantile-normalised inputs let LR model them, while strong L2 keeps it from overfitting. This is the GAM idea behind scikit-learn's SplineTransformer (from prior knowledge of the sklearn docs; no page fetched for this one).
- Result: 0.6139, folds [0.6042, 0.606, 0.6211, 0.6244] -> **discard**. Extra flexibility doesn't help; the effects look close to monotone in quantile space.

### Exp 26: LR branch L2 -> L1 penalty (liblinear, C=0.05)
- Type: exploration
- Hypothesis: Many of the ~180 columns (5 summary stats per aggregate, 93 PCA) are redundant. L1 picks a sparse subset and may generalise better than shrinking all of them (Tibshirani lasso rationale, prior knowledge).
- Result: 0.6134, folds [0.6053, 0.6067, 0.6133, 0.6283] -> **discard**. Folds 1-2 up, fold 3 down a lot. Dense L2 shrinkage suits this diffuse signal better.

### Exp 27: add a RandomForest as a third blend member
- Type: exploration
- Hypothesis: Blends pay off through low correlation ([NVIDIA Grandmasters playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)). A bagged forest with large leaves (min_samples_leaf=30, max_features=0.3) makes different errors from boosting. Weights XGB:LR:RF = 1:2:1. RF handles NaN natively in sklearn 1.6.
- Result: **0.6186** (best 0.6161), folds [0.6059, 0.6094, 0.6265, 0.6327] -> **keep**. All 4 folds up, +0.0025, so the diversity argument holds. Caveat: RF's in-sample training predictions are overconfident, so its z-score std is inflated and its effective weight in the blend is lower than 1. Revisit with OOB predictions.

### Exp 28: RF weight 1 -> 2
- Type: follow-up
- Hypothesis: RF helped on every fold, and its effective weight is damped by in-sample std, so raise its nominal weight.
- Result: 0.6187 (best 0.6186), folds [0.6063, 0.6087, 0.6268, 0.6329] -> **keep** by the strict rule (noise-level; weights look flat here).

### Exp 29: RF max_features 0.3 -> 'sqrt'
- Type: follow-up
- Hypothesis: Fewer candidate features per split (~13 of 172) gives more decorrelated trees and smoother averaged predictions, the classic RF default for classification (Breiman).
- Result: 0.6187 (equal to best), folds [0.6058, 0.6085, 0.6262, 0.6343] -> **discard** (equal is a discard under the rules, though it is 2x faster).

### Exp 30: add ExtraTrees as a fourth member
- Type: exploration
- Hypothesis: Extremely randomised trees (random thresholds) are smoother and differ more from boosting than RF does, adding diversity (NVIDIA playbook diversity argument). min_samples_leaf=30, max_features=0.3, weight 1.
- Result: 0.6181, folds [0.6051, 0.6073, 0.6277, 0.6323] -> **discard**. Too similar to RF; adds nothing.

## Synthesis after 30 experiments
- Best: **0.6187** = z-scored-logit blend of XGB (regularised, weight 1), L2 logistic regression (imputer + quantile-normal, C=0.003, weight 2) and RandomForest (leaf 30, mf 0.3, weight 2). `year` is dropped.
- **Trajectory:** baseline 0.5712 -> regularised XGB 0.6024 -> drop year 0.6035 -> +LR blend 0.6141 -> LR C 0.6154 -> z-blend/weights 0.6161 -> +RF 0.6186.
- **Model-diversity gains are real** (each helped all 4 folds). Tuning inside members (C, L1, splines, RF mf, ExtraTrees) is now at the noise floor.
- **Biggest open area: features for the linear branch.** LR alone is the strongest single model but sees `month`, `hour` and `day_of_month` as linear numbers. Seasonality is non-monotone (gift season), so cyclic or one-hot encodings should help LR without hurting trees.

## Research at exp 30
- An HBS first-purchase study scored new customers on first-transaction variables: number of items, prices, discounts and **whether the purchase fell in a holiday period**. Customers buying more items in the first transaction were more likely to return: [HBS Working Knowledge](https://www.library.hbs.edu/working-knowledge/identify-great-customers-from-their-first-purchase).
- The first basket's *content* (category) is the strongest predictor, with repeat rates of 10-63% by first category: [product journey analysis, 2026](https://mangalaci.github.io/2026/02/19/turning-one-time-buyers-into-returning-customers.html), [Optimove example](https://www.optimove.com/?p=25957). This supports keeping all PCA columns (exps 11/15).
- Planned: (a) cyclic month encoding plus a holiday-season flag; (b) log-count features (trees don't care, but LR goes through the quantile transform anyway, so mainly flags); (c) richer first-basket representation.

### Exp 31: cyclic month encoding (month_sin, month_cos)
- Type: exploration
- Hypothesis: Holiday-period purchases behave differently (HBS). LR can't express a non-monotone month effect from the raw `month` column, but sin/cos can.
- Result: 0.6157, folds [0.6026, 0.6038, 0.6288, 0.6275] -> **discard** (-0.003, 3 of 4 folds down). Likely cause: with only about one year of history, seasonality learned in training is partly confounded with the time trend, and smooth seasonal terms extrapolate badly to the later validation months.

### Exp 32: add k-NN (k=150, on imputed quantile-normal features) as a blend member
- Type: exploration
- Hypothesis: k-NN has the lowest prediction correlation with gradient boosting among common learners ([letsdatascience stacking/blending](https://letsdatascience.com/blog/stacking-and-blending-the-secret-weapons-behind-winning-competitions)), so it may add diversity. A large k keeps it smooth. Weight 1.
- Result: 0.6175, folds [0.6035, 0.6088, 0.628, 0.6297] -> **discard**. Distance in a 170-dim space with diffuse signal is mostly noise, so it dilutes the blend.

### Exp 33: XGB max_depth 3 -> 4 inside the blend
- Type: exploration
- Hypothesis: Now that LR covers the main effects, XGB's job in the blend is interactions, so slightly deeper trees (mcw=60 still guards against small leaves) may add complementary signal even if XGB alone gets worse.
- Result: 0.6188 (best 0.6187), folds [0.6066, 0.6077, 0.6302, 0.6306] -> **keep** by the strict rule (noise-level, 2 of 4 folds up).

### Exp 34: ablation, drop missing indicators from the LR imputer
- Type: ablation / simplification
- Hypothesis: Check whether the NaN-indicator columns matter. If not, the simpler pipeline is preferable.
- Result: 0.6185, folds [0.6064, 0.6076, 0.6301, 0.6301] -> **discard**. Indicators are worth about +0.0003 (noise-level). Kept as-is because the rule needs a strict improvement and simplicity gains here are trivial (one argument).

### Exp 35: RF min_samples_leaf 30 -> 15
- Type: follow-up
- Hypothesis: RF averages many trees, so somewhat smaller leaves may capture more structure without overfitting much.
- Result: 0.6171, folds [0.6059, 0.6086, 0.6284, 0.6256] -> **discard**. Worse, mostly on fold 4. Smaller leaves also make in-sample predictions more overconfident, shrinking RF's effective weight.

### Exp 36: RF min_samples_leaf 30 -> 60
- Type: follow-up (opposite direction)
- Hypothesis: If 15 hurt, larger leaves (smoother RF, less in-sample overconfidence) may help.
- Result: 0.6173, folds [0.6073, 0.6082, 0.6284, 0.6254] -> **discard**. leaf=30 is a local optimum.

## Research after 3 consecutive discards (exps 34-36)
- Under concept drift, weight training samples by exp(-alpha * age) with alpha tuned by CV: [RF concept-drift paper, arXiv 1602.04435](https://arxiv.org/pdf/1602.04435). This is the soft version of training-window selection, with its small-window vs. stale-data trade-off: [non-standard evaluation overview, arXiv 2310.15108](https://arxiv.org/pdf/2310.15108). Learned data importance under drift: [Google Research blog](https://blog.research.google/2024/02/learning-importance-of-training-data.html).
- Relevance: the folds validate on *later* customers, and fold AUCs differ systematically (fold 1 lowest), so customer behaviour likely drifts over time.

### Exp 37: exponential recency sample weights (tau = 12 months)
- Type: exploration
- Hypothesis: Up-weighting recent training customers matches the time-ordered validation (arXiv 1602.04435). A time index t = 12*year + month + day/31 is built from the allowed date-part columns in make_features as a helper column `_t`. ZBlend removes it before fitting or predicting and uses it only to set sample_weight = exp(-(t_max - t)/12), where t_max comes from the fit rows. `year` is still not a model feature.
- Result: 0.6171, folds [0.6069, 0.6103, 0.6303, 0.6209] -> **discard**. Folds 2-3 up slightly, fold 4 down 0.01. With ~13 months of data, tau=12 throws away too much effective sample size (and in fold 4, prior-year same-season rows get down-weighted).

### Exp 38: recency weights with tau = 36 months (milder)
- Type: follow-up
- Hypothesis: A gentler decay (oldest rows about 0.7x) may keep the benefit seen on folds 2-3 without the variance cost.
- Result: 0.6179, folds [0.6069, 0.6084, 0.6292, 0.627] -> **discard**. Recency weighting does not help at any strength tried. Drift is not the bottleneck; dropped.

### Exp 39: z-score RF with out-of-bag predictions instead of in-sample ones
- Type: exploration (blend correctness)
- Hypothesis: RF's in-sample probabilities are overconfident, so its logit std is inflated and its effective weight shrinks. OOB predictions give an honest scale ([sklearn RandomForest oob_score](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), prior knowledge). The weight of 2 then means what it says.
- Result: 0.6187, folds [0.6066, 0.6076, 0.6298, 0.6306] -> **discard** (identical within 0.0001). Blend weighting is flat over a wide range, so the blend is robust and further weight tuning is pointless.

### Exp 40: drop the 13 `*_sum` columns of the prior_* / ratio aggregates
- Type: exploration (feature pruning)
- Hypothesis: Each per-product aggregate comes as min/max/mean/median/sum. The sum is mostly the mean times the number of products, so it duplicates basket size (n_distinct_products) and spreads LR's L2 budget over collinear copies. Dropping it reduces redundancy.
- Result: 0.6181, folds [0.6067, 0.6095, 0.6294, 0.6269] -> **discard**. Feature pruning keeps losing (exps 11, 15, 40). Every block holds some signal and L2/subsampling already handles redundancy.

## Synthesis after 40 experiments
- Best: **0.6188** (exp 33). Since exp 27 (0.6186), nothing has moved beyond noise: 13 experiments on member tuning, blend weights, recency weighting, pruning, seasonal encodings and extra members (ExtraTrees, kNN).
- **Settled:** the XGB + LR + RF blend structure; blend weights (flat); RF leaf size (30 is best); LR penalty (L2, C about 0.003); feature pruning (always hurts); recency weighting (hurts).
- **Not yet tried:** (1) more varied *linear* views, since LR is the strongest member (e.g. a second linear model on a different transform); (2) explicit interaction terms for LR, given that "LR degrades sharply when interactions dominate" ([aimultiple tabular benchmark summary](https://research.aimultiple.com/tabular-models)); (3) XGB learning-rate/rounds now that it is one member of three; (4) quantile resolution.

## Research at exp 40
- Regularised LR matches AutoML on many small tabular datasets; use it as the first choice in low-data settings: [arXiv 2405.07662](https://arxiv.org/pdf/2405.07662.pdf), [PMLBmini arXiv 2409.01635](https://arxiv.org/html/2409.01635v1). This matches exp 24, where LR alone beat XGB alone.
- LR's known weakness is interactions ([aimultiple](https://research.aimultiple.com/tabular-models)). The classic fix is GBDT leaf features fed into LR (He et al. 2014, Facebook ads, prior knowledge). The RF/XGB members cover interactions in the blend already, so the remaining gains are likely small.

### Exp 41: QuantileTransformer n_quantiles 200 -> 1000
- Type: exploration
- Hypothesis: A finer quantile mapping keeps more resolution in heavy-tailed spend/quantity features.
- Result: 0.6188 (equal), folds [0.6068, 0.608, 0.6302, 0.6302] -> **discard**. No effect.

### Exp 42: XGB n_estimators 400 -> 250
- Type: exploration
- Hypothesis: Inside a 3-member blend, a less-fitted (smoother) XGB with depth 4 may be the better complement.
- Result: 0.6189 (best 0.6188), folds [0.6071, 0.6082, 0.6298, 0.6306] -> **keep** by the strict rule (noise-level; also slightly faster).

### Exp 43: XGB n_estimators 250 -> 150
- Type: follow-up
- Hypothesis: Continue in the same direction to see whether an even less-fitted XGB is the better complement.
- Result: 0.6192 (best 0.6189), folds [0.6079, 0.6089, 0.6293, 0.6306] -> **keep** (small; folds 1-2 up).

### Exp 44: XGB n_estimators 150 -> 80
- Type: follow-up
- Hypothesis: Find where under-fitting starts to hurt.
- Result: 0.6186, folds [0.6082, 0.6078, 0.6285, 0.6298] -> **discard**. The optimum is around 150 rounds at lr 0.03.

### Exp 45: XGB max_depth 4 -> 5 (with 150 rounds)
- Type: follow-up (from exp 33)
- Hypothesis: In the blend, XGB's role is interactions. Depth 3->4 helped slightly, so check depth 5 (mcw=60 still caps leaf size).
- Result: 0.6188, folds [0.6076, 0.6087, 0.6293, 0.6295] -> **discard**. Depth 4 stays.

### Exp 46: XGB member boosts from the LR logit (base_margin), i.e. trees fit LR's residuals
- Type: exploration
- Hypothesis: This is the hybrid "linear model + trees on residuals" (the GBDT+LR family, He et al. 2014, prior knowledge). XGB starts from the strong linear signal and only spends its splits on interactions and nonlinearity LR misses. That should give a better member than XGB from scratch. It replaces the plain XGB member, and the separate LR member stays.
- Result: 0.6184, folds [0.6073, 0.6084, 0.6293, 0.6288] -> **discard**. Its predictions now correlate more with the LR member, which costs blend diversity. Independent members work better.

## Research after 3 consecutive discards (exps 44-46)
- Seed averaging is a low-cost variance reduction that helps most for high-variance members: [Scaling Laws/actuarial tabular, arXiv 2609.03106](https://arxiv.org/pdf/2609.03106). XGB with subsample/colsample is seed-sensitive: [stableboost](https://github.com/finite-sample/stableboost). Multi-seed averaging is used for stability in Kaggle practice: [S6E2 write-up](https://pilkwangkim.github.io/posts/S6E2-PRED). Bagging background: [McGill COMP551](https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/16-baggingboosting.pdf).
- Our XGB member uses subsample 0.8 x colsample 0.5 x 0.5, so it is high-variance. Averaging it over seeds should smooth it.

### Exp 47: XGB member averaged over 5 seeds
- Type: exploration
- Hypothesis: Variance reduction of the subsampled XGB member (arXiv 2609.03106, stableboost). Implemented as a small SeedBag wrapper that averages predict_proba over seeds 42..46.
- Result: 0.6190, folds [0.6077, 0.6084, 0.6288, 0.631] -> **discard**. Seed variance isn't the bottleneck: the blend with RF/LR already averages it out.

_EDA (not an experiment), `autoresearch/eda_univariate.py`: univariate AUC per feature on the **earliest 60% of training customers only** (fold-1 training window), so no validation-window labels are used. Findings: `total_spend` alone scores AUC 0.622 (higher spend -> less churn). Next come `prior_median_daily_transactions_max/mean` (~0.59; popular products -> less churn), `total_quantity` 0.59, `prior_median_daily_units_*` ~0.59, and `day_of_month` 0.556. NaN is only ~3.6% (prior_median_daily_* groups). A single feature is about as good as the whole blend on fold 1 (0.608), which suggests the 180-feature LR spreads its L2 budget thin over the few strong signals._

### Exp 48: add a "core" LR member on the 8 strongest basket features, lightly regularised (C=1)
- Type: exploration (EDA-driven)
- Hypothesis: A focused low-dimensional linear model can give the dominant signals (spend, quantity, basket size, product popularity) their full weight without fighting 170 noisy columns for the L2 budget. It is a different "view", so it should add blend diversity. Features: total_spend, total_quantity, n_distinct_products, n_lines, prior_median_daily_transactions_mean/max, prior_median_daily_units_mean/max. Weight 1.
- Result: **0.6224** (best 0.6192), folds [0.612, 0.6105, 0.6294, 0.6376] -> **keep**. +0.0032, the largest gain since exp 27. Folds 1, 2 and 4 up clearly, fold 3 flat. The focused view of the dominant signals works.

### Exp 49: core LR weight 1 -> 2
- Type: follow-up
- Hypothesis: If the focused view is this useful, it may deserve more weight.
- Result: **0.6242** (best 0.6224), folds [0.6141, 0.6113, 0.6292, 0.6422] -> **keep**. 3 of 4 folds up, fold 3 flat.

### Exp 50: core LR weight 2 -> 4
- Type: follow-up
- Hypothesis: Keep following the weight gradient until it turns.
- Result: **0.6253** (best 0.6242), folds [0.6155, 0.6113, 0.6277, 0.6468] -> **keep**. Folds 1 and 4 up, fold 3 slightly down.

## Synthesis after 50 experiments
- Best: **0.6253**. Trajectory: 0.5712 -> 0.6024 (regularised XGB) -> 0.6141 (+LR blend) -> 0.6186 (+RF) -> 0.6192 (XGB tweaks) -> **0.6253 (+core LR on 8 basket features, weight 4)**.
- **Big lesson:** the signal sits in a handful of basket/monetary features (spend, quantity, basket size, product popularity). Models that dilute those among 170 columns underuse them. The jump came from EDA on the earliest training window, not from tuning.
- **Dead ends:** feature pruning of the wide model, recency weights, seasonal encodings, extra tree members (ExtraTrees, kNN), seed bagging, residual boosting, L1, splines.
- **Next:** (1) weight the core member until the gain turns; (2) curate the core feature set (add/remove a few; the EDA ranking is from fold-1 training only); (3) ablation: does the wide LR still help now?

## Research at exp 50
- RFM literature: recency is usually the strongest churn signal, and length/recency/monetary are significant. Validate out-of-time: [Polibatam JAIC study](https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/14108/4028/46078), [MDPI 2022 literature review](https://mdpi.com/2076-3417/12/10/5001). Recency isn't defined at a first purchase, so the **monetary** dimension (total_spend) is the RFM component available here, consistent with the EDA.
- Hybrid regularised-LR + XGB sometimes gives no significant gain over simpler models ([arXiv 1909.11114](https://arxiv.org/pdf/1909.11114)). This supports leaning on simple, focused linear components.

### Exp 51: core LR weight 4 -> 8
- Type: follow-up
- Hypothesis: Check whether the gradient continues. At 8/13 of the weight, the blend is dominated by the core model.
- Result: 0.6256 (best 0.6253), folds [0.6163, 0.6106, 0.6248, 0.6508] -> **keep** by the strict rule. The gain is flattening (+0.0003), and fold 3 keeps dropping (0.6294 -> 0.6248) while fold 4 rises.

### Exp 52: ablation, core LR alone (weights 0:0:0:1)
- Type: ablation
- Hypothesis: Measure whether the other three members still add anything. If the core alone is close, the extra machinery is hard to justify (program.md simplicity rule).
- Result: 0.6230, folds [0.6158, 0.6069, 0.6158, 0.6533] -> **discard**. Notable: **an 8-feature logistic regression alone scores 0.6230**, beating every model up to exp 47 including the tuned 3-member blend (0.6192). The other members add +0.0026, mostly on fold 3 (0.6158 -> 0.6248), so they stay. For the comparison write-up: almost all the gain comes from finding the right few features.

### Exp 53: core features + 3 more basket aggregates (qty_share_of_prior_units_sum, prior_median_daily_transactions_sum, prior_median_daily_units_sum)
- Type: follow-up (curating the core)
- Hypothesis: These were next in the fold-1-window univariate ranking (AUC 0.57-0.59). Adding them may strengthen the core view, unless they are redundant with what is already there.
- Result: 0.6256 (equal), folds [0.616, 0.6104, 0.6246, 0.6514] -> **discard**. Redundant: the *_sum stats are mean x basket size, which the core already has.

### Exp 54: core + avg_unit_price and day_of_month
- Type: follow-up (curating the core)
- Hypothesis: Add the non-redundant directions. Unit price (premium vs. cheap items; LR on quantile features can't form the spend/quantity ratio) and day_of_month (univariate 0.556 on the fold-1 window, + direction; possibly payday or month-end wholesale timing).
- Result: **0.6280** (best 0.6256), folds [0.6185, 0.6129, 0.6278, 0.6529] -> **keep**. All 4 folds up, +0.0024.

### Exp 55: core + prior_transactions_mean, prior_units_min
- Type: follow-up (curating the core)
- Hypothesis: These product-history stats have the *opposite* sign to the median-daily ones (+ churn direction, univariate ~0.55). The raw transaction counts behind the basket's products are a different popularity notion from daily medians, so they may add a non-redundant direction.
- Result: 0.6250, folds [0.6163, 0.6111, 0.6263, 0.6462] -> **discard**. All folds down. The positive-direction counts don't transfer to later customers.

### Exp 56: core LR C 1.0 -> 0.1
- Type: exploration
- Hypothesis: With 10 correlated features (spend/quantity/lines), mild L2 shrinkage may stabilise the coefficients across folds.
- Result: 0.6258, folds [0.6168, 0.6112, 0.6253, 0.6499] -> **discard**. All folds down. With ~2.4k+ rows and 10 features the core is not overfitting, and shrinkage biases the strong spend coefficient.

### Exp 57: core + the 3 strongest PCA components (pca_17, pca_46, pca_01)
- Type: follow-up (curating the core)
- Hypothesis: First-basket *content* predicts return ([product journey analysis](https://mangalaci.github.io/2026/02/19/turning-one-time-buyers-into-returning-customers.html)). These are the description-embedding directions with the highest fold-1-window univariate AUC (0.53-0.55).
- Result: 0.6217, folds [0.6155, 0.609, 0.6232, 0.6389] -> **discard**. All folds down, fold 4 by 0.014. Individual PCA directions are unstable over time, so content is better left to the wide models.

## Research after 3 consecutive discards (exps 55-57)
- Log is the standard tail compressor for positive heavy-tailed data ([Frees, fat-tailed regression ch. 17](https://users.ssc.wisc.edu/~ewfrees/ActuarialRegression2009/C17Fat.html), [log transforms for skewed features](https://theneuralbase.com/lightgbm/learn/intermediate/log-transforms-for-skewed-features/)). The quantile-normal transform is the alternative and the choice is data-dependent ([MLMastery quantile transforms](https://machinelearningmastery.com/quantile-transforms-for-machine-learning/), [quantile-matching likelihood, arXiv 2001.03709](https://arxiv.org/pdf/2001.03709)). Recommendation: compare log1p, Yeo-Johnson and quantile empirically on time-based validation.
- Relevance: the quantile-normal map **caps the extremes** (huge wholesale baskets get squashed to about +3 sd), whereas log keeps the ordering *and* relative distance in the tail. For the core LR, where spend is the main signal, that can matter.

### Exp 58: core LR uses a Yeo-Johnson PowerTransformer (standardised) instead of QuantileTransformer
- Type: exploration
- Hypothesis: A parametric log-like transform keeps the tail information that the quantile map discards (research above). Yeo-Johnson handles zeros and negatives and is fitted on X_fit only.
- Result: 0.6275 (best 0.6280), folds [0.6184, 0.6129, 0.6275, 0.6514] -> **discard**. Practically identical, so the tail treatment doesn't matter. The quantile transform stays.

### Exp 59: core minus n_lines
- Type: ablation / simplification
- Hypothesis: n_lines is about n_distinct_products (most lines are distinct products), so it may be redundant. Removing it is simpler and may cut variance.
- Result: 0.6260, folds [0.6171, 0.6109, 0.6251, 0.6511] -> **discard**. All folds down, so n_lines carries something (repeated lines of the same product may signal reseller behaviour).

### Exp 60: core + hour
- Type: follow-up (curating the core)
- Hypothesis: Purchase hour separates business buyers (office hours) from consumers (evenings). Wholesalers reorder, so hour is a non-redundant direction like day_of_month was.
- Result: 0.6282 (best 0.6280), folds [0.6185, 0.613, 0.6281, 0.6531] -> **keep** by the strict rule (noise-level).

## Synthesis after 60 experiments
- Best: **0.6282**. Since exp 48 the core-LR idea has taken the score from 0.6192 to 0.6282. Curating the core helped twice (avg_unit_price + day_of_month: +0.0024; hour: +0.0002). It failed with redundant sums, prior counts, PCA directions, removing n_lines, shrinkage (C=0.1) and Yeo-Johnson.
- **Pattern:** the core model wants *non-redundant basket-shape directions*. Directions that just repeat basket size do nothing, and unstable ones (PCA, prior counts) hurt.
- **Next:** wholesaler/reseller shape (units per line, spend per line) as explicit ratios for the core, since LR can't form ratios from separately quantile-mapped columns.

## Research at exp 60
- The retailer sells mainly gift items and **many customers are wholesalers**. Revenue is concentrated (top 10% of customers about 50% of revenue): [OpenML Online Retail description](https://api.openml.org/d/43368), [TowardsAI segmentation write-up](https://pub.towardsai.net/customer-segmentation-and-retention-strategy-using-transactional-data-9b74e0d27e35), [RFM with Polars](https://github.com/Agungvpzz/RFM-Analysis-with-Polars).
- Suggested reseller signals: units per invoice line, average unit price, units per invoice. Unit price is already in the core; units-per-line is not.

### Exp 61: core + units_per_line and spend_per_line (ratios built in make_features)
- Type: follow-up (research-driven)
- Hypothesis: Resellers buy many units per line at low unit prices and come back to restock. LR on separately quantile-mapped totals can't express this ratio. In exp 14 these ratios didn't help the *wide* models (trees can approximate them), but the core LR may benefit.
- Result: 0.6277, folds [0.6183, 0.6127, 0.6261, 0.6539] -> **discard**. Noise-level. In quantile space, spend - lines is already close to the log ratio, so LR could approximate it.

### Exp 62: ablation, drop the wide LR member (weight 2 -> 0)
- Type: ablation
- Hypothesis: The core LR may now cover what the wide LR contributed. If the score holds, the blend is simpler.
- Result: 0.6272, folds [0.6182, 0.6119, 0.6243, 0.6544] -> **discard**. The wide LR still adds about +0.001 (mostly fold 3), so it stays.

### Exp 63: add a "core XGB" member (trees on the 11 core features only)
- Type: exploration
- Hypothesis: The core LR showed that focusing on strong features helps. A shallow XGB restricted to the same features can find *interactions among the strong signals* (e.g. spend x unit price = wholesaler) without wasting splits on 160 noisy columns. Depth 3, 200 rounds, lr 0.03, mcw 60, weight 2.
- Result: 0.6266, folds [0.6174, 0.612, 0.6257, 0.6515] -> **discard**. Interactions among the core features found by trees don't transfer forward in time.

## Research after 3 consecutive discards (exps 61-63)
- Hierarchical interaction lasso (an interaction enters only with its main effects): [Bien, Taylor & Tibshirani, arXiv 1205.5050](https://arxiv.org/pdf/1205.5050), [hierNet.logistic](https://rdrr.io/cran/hierNet/man/hierNet.logistic.html). glinternet (Lim & Hastie 2015) for strong hierarchy: [PMC4706754](https://pmc.ncbi.nlm.nih.gov/articles/PMC4706754). Main-effects-first is the standard. Whether interactions raise AUC is an empirical question.
- After exp 63 (tree interactions on the core didn't help), a smooth multiplicative interaction in the LR is the last cheap interaction test. The hierarchical penalty isn't available in sklearn, so as an approximation the products of quantile-normal core features go in a *separate* member, with main effects included, so the main-effects core LR stays untouched.

### Exp 64: add a core LR with pairwise interactions (PolynomialFeatures degree 2, interaction_only) as a member
- Type: exploration
- Hypothesis: Smooth pairwise products (e.g. spend x unit price, spend x hour) may carry the wholesaler/consumer interaction more stably than tree splits did in exp 63 (hierNet rationale). C=0.05 shrinks the 55 product terms. Weight 2.
- Result: 0.6288 (best 0.6282), folds [0.6183, 0.6127, 0.6273, 0.6571] -> **discard on simplicity grounds**, overriding the mechanical keep. The whole +0.0006 comes from fold 4 (+0.004), while folds 1-3 are all slightly *down*. program.md says gains under ~0.002 are likely noise, to prefer changes that help most folds, and not to keep small gains that add code (this is a 10-line member). train.py was reverted by hand and best_train.py re-synced, with a re-check run confirming 0.6282 (not counted as an experiment).
- _Note: LR fits emit spurious `RuntimeWarning: divide by zero/overflow in matmul` from numpy's macOS Accelerate BLAS (known sklearn/numpy 2 issue). Results are finite and deterministic, so these are ignored._

### Exp 65: core LR class_weight='balanced'
- Type: exploration
- Hypothesis: Churn is 41-45%, so the classes are only mildly imbalanced. Balanced weights change the LR's decision geometry slightly and could shift the core coefficients. A cheap check (one argument).
- Result: 0.6283 (best 0.6282), folds [0.6185, 0.613, 0.6282, 0.6534] -> **keep** by the strict rule (noise-level, no code cost).

### Exp 66: core + is_international flag (any country_* dummy set; UK is the omitted level)
- Type: follow-up (curating the core; research-driven)
- Hypothesis: Many customers of this UK gift retailer are wholesalers ([OpenML description](https://api.openml.org/d/43368)), and foreign buyers ordering from a UK online gift wholesaler are disproportionately resellers. One aggregated flag is more stable than 13 sparse country dummies (each under 2%).
- Result: 0.6286 (best 0.6283), folds [0.6155, 0.612, 0.6298, 0.6572] -> **discard on simplicity grounds** (same standard as exp 64). +0.0003 is noise, folds 1-2 down and 3-4 up, and it adds a make_features block. Reverted by hand; a re-check run gives 0.6283 (not counted).

### Exp 67: wide LR C 0.003 -> 0.01
- Type: follow-up (re-tune after the structure change)
- Hypothesis: C=0.003 was tuned when the wide LR was the main linear member. Now that the core LR carries the main effects, the wide LR's job is the long tail of weak features, and a different penalty may suit that.
- Result: **0.6290** (best 0.6283), folds [0.6186, 0.6133, 0.6293, 0.6546] -> **keep**. All 4 folds up slightly, no code cost.

### Exp 68: wide LR C 0.01 -> 0.03
- Type: follow-up
- Hypothesis: Continue the direction.
- Result: 0.6293 (best 0.6290), folds [0.6188, 0.6137, 0.6295, 0.6552] -> **keep**. All folds up slightly again.

### Exp 69: wide LR C 0.03 -> 0.1
- Type: follow-up
- Hypothesis: Find where the C gradient turns.
- Result: 0.6294 (best 0.6293), folds [0.6189, 0.6134, 0.6297, 0.6554] -> **keep** (flattening; +0.0001).

### Exp 70: wide LR blend weight 2 -> 4
- Type: follow-up
- Hypothesis: A less-shrunk wide LR is a stronger member and may deserve more weight.
- Result: 0.6296 (best 0.6294), folds [0.6181, 0.6133, 0.6323, 0.6547] -> **keep** (fold 3 up, others flat).

## Synthesis after 70 experiments
- Best: **0.6296**. Exps 60-70: core + hour (+0.0002), core LR balanced (+0.0001), wide LR C 0.003 -> 0.1 (+0.0011, all folds up each step), wide LR weight 4 (+0.0002). Re-tuning the wide LR *after* the core existed was worthwhile: its role changed from main-effects model to long-tail model.
- Discarded on simplicity grounds despite a nominal gain: exp 64 (pairwise core LR) and exp 66 (is_international). Both were one-sided fold patterns with added code.
- Still dead: tree models restricted to the core (exp 63), extra core features beyond the 11.
- Remaining budget: 30. Plan: holiday/seasonal signal in the core (research below); re-check XGB/RF settings under the new blend; final simplification ablations near the end.

## Research at exp 70
- Temporal features can carry the most information in purchase prediction ([arXiv 1512.04912, Portrait of an Online Shopper](https://arxiv.org/pdf/1512.04912.pdf)). Festival-period behaviour was among the dominant features for repeat buyers ([Erasmus Double-11 thesis](https://thesis.eur.nl/pub/72505/Thesis-Draft-Ver3.0.pdf), [U-Tokyo Singles' Day study](https://must.c.u-tokyo.ac.jp/sigam33/sigam3305.pdf)). Holiday timing of the first purchase was in the HBS model ([HBS](https://www.library.hbs.edu/working-knowledge/identify-great-customers-from-their-first-purchase)).
- Exp 31 (cyclic month in *all* models) hurt. The core LR is a cleaner place to test a seasonal term, as a single linear month.

### Exp 71: core + month
- Type: follow-up (research-driven)
- Hypothesis: Holiday-season first purchases (gift buyers) are less likely to repeat (HBS, Double-11 thesis). A linear month term in the focused core captures the Q4-vs-rest gradient (months run 1-12, with Q4 at the top).
- Result: 0.6293, folds [0.6181, 0.6134, 0.6321, 0.6538] -> **discard**. Seasonality doesn't transfer across the time folds (consistent with exps 13 and 31). With one year of history, month is confounded with cohort and time.

### Exp 72: ablation, XGB member weight 1 -> 0
- Type: ablation
- Hypothesis: XGB has the smallest weight and is the most similar to RF. If removing it doesn't hurt, it's a simplification candidate for the end.
- Result: **0.6298** (best 0.6296), folds [0.6182, 0.613, 0.6318, 0.6561] -> **keep**. **XGBoost no longer contributes.** Once the linear members are strong, RF covers the nonlinear part. A notable finding for the benchmark: the best autoresearch model does not need XGBoost at all.

### Exp 73: remove the XGB member code entirely (simplification)
- Type: simplification
- Hypothesis: With weight 0 the XGB member only costs code and runtime. Removing it (plus the XGB_PARAMS block and the xgboost import) should give an identical score. program.md says "equal AUC with less code is a win", so an identical score counts as a **keep** here, overriding the helper's strict ">" rule.
- Result: 0.6298 (equal), identical folds [0.6182, 0.613, 0.6318, 0.6561], 6.9s vs 8.0s -> **keep (simplification)**. The helper restored the old file on the tie. The candidate was saved beforehand and promoted to train.py/best_train.py by hand; a re-check run gives 0.6298. train.py no longer imports xgboost.

### Exp 74: RF blend weight 2 -> 4
- Type: follow-up
- Hypothesis: With XGB gone, RF is the only nonlinear member and may deserve more weight.
- Result: 0.6292, folds [0.6179, 0.6127, 0.6324, 0.6537] -> **discard**.

### Exp 75: RF blend weight 2 -> 1
- Type: follow-up (opposite direction)
- Hypothesis: If more RF hurt, less RF may help, since the linear members carry the signal.
- Result: **0.6300** (best 0.6298), folds [0.6182, 0.613, 0.6313, 0.6574] -> **keep** (noise-level; no code cost).

### Exp 76: ablation, RF weight 1 -> 0 (purely linear: wide LR + core LR)
- Type: ablation
- Hypothesis: Check whether any tree model is still needed. If two logistic regressions match, that is a striking simplification.
- Result: 0.6299 (best 0.6300), folds [0.6182, 0.613, 0.6304, 0.6579] -> **discard** by the strict rule, but RF now adds only **+0.0001**, far below the ~0.002 noise floor.

### Exp 77: remove the RF member code entirely (two logistic regressions only)
- Type: simplification
- Hypothesis: program.md: "Simpler is better ... equal AUC with less code is a win", and gains under ~0.002 are noise. Dropping RF (300 trees) removes a model family, cuts runtime, and leaves a fully linear, interpretable model. The expected score is 0.6299 vs 0.6300, which I treat as equal and therefore **keep as a simplification**. This is a deliberate judgment call, recorded here. From now on `run_experiment.py` compares against the latest keep rather than the max, so later experiments are judged against what best_train.py actually holds.
- Result: 0.6299, folds [0.6182, 0.613, 0.6304, 0.6579], **1.1s** (vs 7.1s) -> **keep (simplification)**. Promoted by hand (the helper restored on the strict rule). Re-check run: 0.6299. **The model is now two logistic regressions:** a wide one on all quantile-normalised features (C=0.1) and a focused one on 11 basket/timing features (C=1, balanced), blended 1:2 on z-scored logits. No tree models remain.

### Exp 78: blend weights wide:core 1:2 -> 1:3
- Type: follow-up (re-tune after removing RF)
- Hypothesis: The weight optimum may have shifted now that RF is gone.
- Result: **0.6305** (best 0.6299), folds [0.6193, 0.6129, 0.6299, 0.6601] -> **keep**. Folds 1 and 4 up, 2 and 3 flat to slightly down.

### Exp 79: blend weights 1:3 -> 1:5
- Type: follow-up
- Hypothesis: Continue the direction.
- Result: 0.6305 (equal), folds [0.62, 0.6135, 0.6276, 0.661] -> **discard**. Plateau, with fold 3 dropping as the core dominates.

### Exp 80: the wide LR excludes the core features (orthogonal views)
- Type: exploration
- Hypothesis: Today the core features appear in both members, so the wide LR spends part of its L2 budget re-estimating them. Restricting the wide LR to the *remaining* ~160 columns makes the two views complementary: the core LR models the main effects, the wide LR only the long tail. This keeps the ColumnTransformer pattern already in use (remainder='drop' replaced by 'passthrough' on the complement).
- Result: 0.6305 (equal), folds [0.6205, 0.6147, 0.629, 0.6578] -> **discard** (equal, no simpler).

## Synthesis after 80 experiments
- Best: **0.6305**, from two logistic regressions on quantile-normalised features: a wide one (all 172 columns + NaN indicators, C=0.1) and a focused core one (11 basket/timing features, C=1, balanced), blended 1:3 on z-scored logits. CV time is 1.1s, down from 8s with the tree ensemble.
- **Arc of the run:** regularised XGB (0.602) -> + linear blend (0.614) -> + RF (0.619) -> + EDA-driven core LR (0.625) -> core curation and re-tuning (0.630) -> **ablations showed XGB and RF no longer contribute**, so they were removed (simplicity rule).
- **Lesson:** with weak signal and time drift, the winning approach is a well-specified linear model that gives the few strong features low shrinkage and all the rest high shrinkage.

## Research at exp 80
- glmnet's `penalty.factor` gives each coefficient its own penalty multiplier: small for features believed important, large for the rest ([glmnet docs](https://rdrr.io/cran/glmnet/man/glmnet.html), [r-bloggers deep dive](https://www.r-bloggers.com/2018/11/a-deep-dive-into-glmnet-penalty-factor/)). xtune learns feature-specific shrinkage from external information and reports better accuracy from differential shrinkage ([xtune README](https://rdrr.io/cran/xtune/f/README.md)). Pretraining-lasso uses the same idea: [arXiv 2401.12911](https://arxiv.org/pdf/2401.12911). sklearn lacks penalty factors ([sklearn issue #11566](https://github.com/scikit-learn/scikit-learn/issues/11566)), **but for L2 on standardised inputs, multiplying a column by k is equivalent to a penalty factor of 1/k^2**.
- This is exactly what the 2-model blend approximates. A single LR with up-scaled core columns could replace the blend and the ZBlend class.

### Exp 81: one LR with differential shrinkage (core columns x3 after the quantile transform) replaces the 2-model blend
- Type: simplification / exploration (research-driven)
- Hypothesis: Scaling the 11 core columns by k=3 after quantile-normalising gives them a penalty factor of 1/9 (glmnet penalty.factor equivalence). One model then does what the blend does, and ZBlend (~25 lines) goes away. C=0.1 as in the wide LR.
- Result: 0.6051, folds [0.5912, 0.5937, 0.6152, 0.6201] -> **discard** (-0.025). Flaw in the setup: in the blend, the wide LR at C=0.1 is *down-weighted* 1:3, so its many weak coefficients barely matter. In a single model at C=0.1 all ~170 weak coefficients enter at full strength and overfit. The penalty must be strong for the bulk (C ~0.003, as in exp 19) and weak for the core.

### Exp 82: single LR, C=0.003 for the bulk, core scaled x18 (effective core C ~ 0.003 x 18^2 ~ 1)
- Type: follow-up (fix of exp 81)
- Hypothesis: Same glmnet penalty.factor idea with sensible magnitudes. The bulk gets the strong shrinkage that worked in exp 19 and the core gets the light shrinkage that worked for the core LR (C=1).
- Result: 0.6199, folds [0.609, 0.6071, 0.6265, 0.637] -> **discard** (-0.011). Better than exp 81 but clearly below the blend. In a *joint* fit, the weakly penalised core coefficients and the many bulk coefficients compete for the same signal, and correlated bulk columns pull weight away from the core. Fitting the two views *separately* and blending gives a more stable estimate. The blend stays and the idea is dropped.

### Exp 83: core LR without class_weight='balanced' (simplification)
- Type: ablation / simplification
- Hypothesis: 'balanced' added only +0.0001 (exp 65). If removing it is equal or better, the default is simpler.
- Result: 0.6305 (equal), folds [0.6193, 0.613, 0.6295, 0.6602] -> **keep (simplification)**, promoted by hand. Re-check: 0.6305.

### Exp 84: keep `year` (make_features becomes the identity)
- Type: ablation / simplification
- Hypothesis: Dropping `year` helped the XGB model (exp 12), but the model is now purely linear. If keeping it is equal or better, make_features can become a no-op.
- Result: **0.6313** (best 0.6305), folds [0.6197, 0.614, 0.6298, 0.6616] -> **keep**. All 4 folds up, and simpler. In the linear model, `year` acts as a mild cohort shift, whereas trees overfit their split on it.

### Exp 85: wide LR imputer without missing indicators (simplification)
- Type: ablation / simplification
- Hypothesis: The NaN indicators were worth about +0.0003 in the old tree blend (exp 34). Re-check them in the linear-only model.
- Result: 0.6314 (best 0.6313), folds [0.6199, 0.6138, 0.63, 0.6618] -> **keep** (simpler, equal or better).

### Exp 86: core LR imputer without missing indicators (simplification)
- Type: ablation / simplification
- Hypothesis: Same check for the core member (4 of its 11 features have about 3.6% NaN).
- Result: **0.6328** (best 0.6314), folds [0.6209, 0.6149, 0.631, 0.6645] -> **keep**. All 4 folds up. The unpenalised (C=1) indicator coefficients were fitting noise.

### Exp 87: wide LR C 0.1 -> 0.3
- Type: follow-up (re-tune after the simplifications)
- Hypothesis: The wide LR C gradient was still slightly positive at 0.1 (exp 69).
- Result: 0.6329 (best 0.6328), folds [0.6206, 0.615, 0.6314, 0.6647] -> **keep** by the strict rule (noise-level, no code cost).

### Exp 88: core LR C 1 -> 10
- Type: follow-up
- Hypothesis: C=0.1 hurt the core (exp 56), so less shrinkage may help a little further.
- Result: **0.6334** (best 0.6329), folds [0.6214, 0.6159, 0.6317, 0.6645] -> **keep**. 3 of 4 folds up.

### Exp 89: core LR unpenalised (penalty=None)
- Type: follow-up
- Hypothesis: With 11 features and 2,400+ rows, the core needs no shrinkage at all. If penalty=None matches, the core is a plain maximum-likelihood logistic regression, which is the most interpretable form.
- Result: 0.6334 (equal), folds [0.6216, 0.616, 0.6317, 0.6642] -> **discard** (equal and no simpler in code; C=10 is effectively unpenalised already).

### Exp 90: blend weights wide:core 1:3 -> 1:2
- Type: follow-up (re-tune after the simplifications)
- Hypothesis: The wide LR changed (year, no indicators, C=0.3), so the weight optimum may have moved.
- Result: 0.6332, folds [0.6206, 0.6155, 0.6331, 0.6635] -> **discard**. 1:3 stays.

## Synthesis after 90 experiments
- Best: **0.6334** (exp 88). Exps 81-90: the single-LR penalty-factor idea failed (exps 81-82; joint fitting is worse than blending separate views). Then simplification ablations *gained* AUC: removing class_weight (equal), keeping `year` (+0.0008), dropping NaN indicators from both members (+0.0001, +0.0014), and re-tuning C (wide 0.3, core 10; +0.0006). Removing things that had been tuned for the tree ensemble helped the linear model.
- The final structure is two logistic regressions: wide (all columns, median-impute, quantile-normal, C=0.3) and core (11 basket/timing features, same preprocessing, C=10), blended 1:3 on z-scored logits.

## Research at exp 90
- Logistic regression ensembles over *feature subsets* (random subspace method, Lorens, ELR, Super RaSE) combine LRs fitted on different variable subsets by averaging: [Logistic Ensemble Models, arXiv 1806.04555](https://arxiv.org/pdf/1806.04555), [Super RaSE (Zhu & Feng 2021)](https://mail.yangfeng.hosting.nyu.edu/publication/zhu-2021-super/zhu-2021-super.pdf), [random subspacing for regression ensembles](https://edepositireland.ie/handle/2262/13279). Simple averaging is the safer default at small n.
- The current model is a 2-subset LR ensemble (all features + core). A third, semantically defined subset (the 93 product-content PCA columns) is the natural next view.

### Exp 91: add a third LR member on the pca_* columns only (product-content view), weight 1
- Type: exploration (research-driven)
- Hypothesis: Subset LR ensembles (Lorens/ELR) gain from diverse views. A content-only LR (C=0.01, since there are 93 noisy directions) gives first-basket content its own vote ([product journey analysis](https://mangalaci.github.io/2026/02/19/turning-one-time-buyers-into-returning-customers.html)) instead of being drowned out in the wide LR.
- Result: 0.6308, folds [0.6203, 0.6162, 0.6309, 0.6557] -> **discard**. Fold 4 drops 0.009: content directions are not stable over time (consistent with exps 15 and 57).

## Research after 3 consecutive discards (exps 89-91)
- QuantileTransformer maps each column through its empirical CDF to the normal quantile function. Values at or beyond the fitted range go to the distribution bounds ([sklearn QuantileTransformer docs, 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.QuantileTransformer.html)). For a **0/1 one-hot column** the CDF has only two steps, so the dummies land far in the tails (about +/-5 sd), instead of the bounded 0/1 the L2 penalty is calibrated for. A rare country dummy then gets very high leverage in the wide LR. The docs don't discuss binary columns explicitly; this is my inference from the documented mapping.

### Exp 92: wide LR passes the 30 one-hot columns (weekday_*, time_of_day_*, country_*) through untransformed
- Type: exploration (research-driven)
- Hypothesis: Quantile-normalising binary dummies blows them up to extreme values (research above). Leaving them as 0/1 restores sensible leverage and penalty scaling. Only the continuous columns get imputed and quantile-normalised (ColumnTransformer; nothing learned on validation rows).
- Result: 0.6329 (best 0.6334), folds [0.6215, 0.616, 0.631, 0.6629] -> **discard**. No gain. At C=0.3 with 4k rows, the dummies' scale isn't a binding problem, and the wide member has only 1/4 of the blend weight.

### Exp 93: core + year
- Type: follow-up (from exp 84)
- Hypothesis: Keeping `year` helped the wide LR (+0.0008, all folds). In the lightly penalised core LR, a cohort shift term may help more directly.
- Result: 0.6320, folds [0.6212, 0.6154, 0.6306, 0.661] -> **discard**. Unpenalised, year overfits the cohort shift. Lightly shrunk in the wide LR it's fine.

## Research after 3 consecutive discards (exps 91-93)
- Adaptive data analysis: reusing the same holdout or CV across many adaptive choices overfits it ([Dwork et al. reusable holdout, KDnuggets summary](https://www.kdnuggets.com/2015/08/feldman-avoid-overfitting-holdout-adaptive-data-analysis.html/2), [arXiv 1506.02629](https://arxiv.org/abs/1506.02629v2)). HPO with repeated CV shows significant adaptive overfitting vs. a truly unseen test set ([Leiden thesis 2023-24](https://theses.liacs.nl/pdf/2023-2024-SchrderSSietse.pdf)). Description-length style guarantees favour simpler selected models ([Rip van Winkle's Razor, arXiv 2102.13189](https://arxiv.org/pdf/2102.13189)).
- **Implication for this run:** after ~93 adaptive choices on the same 4 folds, the many +0.0001-0.0005 keeps are partly CV-overfit. The held-out test (run by the human) will likely come in somewhat below 0.633. For the last 7 experiments I'll favour **ablations that test robustness or simplify** over further micro-tuning, and report the expected optimism.

### Exp 94: ablation, core minus `hour`
- Type: ablation / simplification
- Hypothesis: `hour` was added for +0.0002 (exp 60, noise-level). If removing it is equal or better, it goes (adaptive-overfitting caution).
- Result: 0.6330 (best 0.6334), folds [0.6212, 0.6157, 0.6318, 0.6634] -> **discard**. `hour` is worth about +0.0004 (all folds slightly). It stays; it is one list entry, so not a meaningful complexity cost.

### Exp 95: ablation, core minus `day_of_month`
- Type: ablation (attribution)
- Hypothesis: Exp 54 added avg_unit_price and day_of_month together (+0.0024). Attribute the gain. If day_of_month is not pulling its weight, a timing feature with a weak causal story should go.
- Result: 0.6286 (best 0.6334), folds [0.6181, 0.6125, 0.6268, 0.6572] -> **discard**. **day_of_month is worth about +0.005, on all 4 folds.** Later-in-month first purchases churn more, and the effect holds across every time-ordered fold. **Flag for the human:** the causal story is weak (month-end business ordering is plausible). It is worth checking whether day_of_month interacts with how the 90-day churn window or the data extraction was defined, since a label-construction artifact would also produce this pattern. I can't check that without looking at data-prep code outside my allowed scope.

### Exp 96: ablation, core minus `avg_unit_price`
- Type: ablation (attribution)
- Hypothesis: The other half of the exp-54 pair.
- Result: 0.6328 (best 0.6334), folds [0.6208, 0.6151, 0.6314, 0.664] -> **discard**. avg_unit_price is worth only about +0.0006 now; day_of_month carried nearly all of the exp-54 gain.

## Research after 3 consecutive discards (exps 94-96)
- Context for the day_of_month effect (exp 95): **UK/EU online revenue runs about 14% above the daily average at month-end** (payday economy) ([Wayflyer 2026](https://wayflyer.com/data/articles/20260617-payday-economy)). Spending jumps on payday ([InMarket](https://inmarket.com/report-consumer-spending-increases-33-on-payday/), [DMA](https://dma.org.uk/article/should-advertisers-pay-more-attention-to-payday)), and spending is excessively sensitive to income timing ([AEA 2017 paper](https://topcat.aeaweb.org/conference/2017/preliminary/paper/a7SdibQd), [Vellekoop SSRN](https://publikationen.ub.uni-frankfurt.de/opus4/files/47634/SSRN-id3296822.pdf)). A plausible mechanism: late-month first purchases in a UK shop skew toward payday-driven one-off consumer buys, which repeat less than planned business orders. This makes the feature more credible, but the label-artifact check flagged in exp 95 still stands.

### Exp 97: replace custom ZBlend with sklearn VotingClassifier(soft) (simplification)
- Type: simplification
- Hypothesis: ZBlend (~25 lines) was introduced (exp 22) to stop compressed, heavily-regularised LR probabilities from losing weight in a probability average. Both members are now lightly regularised (C=0.3, C=10), so their probability scales are similar and a plain soft vote with weights [1, 3] may match. That would replace the custom class with a built-in.
- Result: 0.6333 (best 0.6334), folds [0.6212, 0.6163, 0.6324, 0.6634] -> **keep (simplification)**, the same judgment as exp 77: -0.0001 is far inside the noise floor, and it removes the 25-line custom ZBlend class in favour of a standard sklearn component. Promoted by hand; re-check gives 0.6333. **The current best (best_train.py) is 0.6333.**

### Exp 98: blend weights 1:3 -> 1:4 (VotingClassifier)
- Type: follow-up
- Hypothesis: The probability-average blend weights differently from the z-blend, so the optimum may sit at a slightly different ratio.
- Result: **0.6335** (best 0.6333), folds [0.6218, 0.6163, 0.6313, 0.6644] -> **keep** (noise-level, no code cost).

### Exp 99: blend weights 1:4 -> 1:6
- Type: follow-up
- Hypothesis: Check whether the weight gradient continues.
- Result: 0.6330, folds [0.6223, 0.6162, 0.6298, 0.6639] -> **discard**. 1:4 is the optimum.

### Exp 100: wide LR C 0.3 -> 1.0
- Type: follow-up
- Hypothesis: The last unexplored step on the wide-LR penalty path (the gradient was flat-positive up to 0.3).
- Result: 0.6334 (best 0.6335), folds [0.6217, 0.6163, 0.6313, 0.6645] -> **discard**. Plateau.

## Final summary (100 experiments)
- **Best CV AUC: 0.6335** (exp 98). Baseline was 0.5712 (+0.062). The mean of the 4 time-ordered folds is [0.6218, 0.6163, 0.6313, 0.6644], and all four folds beat the baseline by +0.04 to +0.10.
- **Final model (train.py = best_train.py):** a soft-voting blend (weights 1:4) of two logistic regressions with median imputation and a quantile-to-normal transform fitted on training rows only:
  1. *wide* LR on all 173 columns, C=0.3;
  2. *core* LR on 11 hand-picked basket/timing features (total_spend, total_quantity, n_distinct_products, n_lines, prior_median_daily_transactions_mean/max, prior_median_daily_units_mean/max, avg_unit_price, day_of_month, hour), C=10.
  No feature engineering (make_features is the identity), no tree models, no xgboost import. CV runs in 0.7s.
- **What mattered, in order of gain:** regularising XGB (+0.026); blending in a regularised LR (+0.010); the EDA-driven focused core LR (+0.003, then curation: day_of_month +0.005); RF diversity (+0.0025, later made redundant); removing NaN indicators / keeping year / re-tuning C in the linear-only model (+0.003 total). Ablations then showed XGB and RF contributed nothing once the linear views were good, so they were removed under the simplicity rule.
- **Judgment calls (all logged):** exps 64 and 66 were discarded despite a nominal +0.0003-0.0006 (one-sided folds, added code). Exps 73, 77, 83 and 97 were kept as simplifications at equal or -0.0001 AUC. Exp 20 was a no-op (failed edit) and is counted.
- **Caveats for the human:** (1) about 100 adaptive choices on the same 4 folds means the CV is optimistic (adaptive overfitting, research after exp 93). Expect the held-out test AUC to be lower, and treat sub-0.002 differences as noise. (2) The day_of_month effect (+0.005) is strong and consistent but has a weak causal story (payday effect is plausible); worth checking that it isn't a label-construction artifact. (3) The univariate EDA (before exp 48) used only the earliest 60% of customers, i.e. fold 1's training window. Those rows are never validation rows in any fold (every fold validates on later customers), so feature selection used no validation labels.
