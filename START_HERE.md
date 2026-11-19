# Member B — Candidate models, training and Bayesian optimization

Read this file first, then `CONTRACT_AND_INTEGRATION.md`. Your archive contains original assigned source/test files under `owned/`; implementation tasks below remain to be completed. Work against the full original scaffold, not this partial archive alone.

Branch: `member-b/models-training`

## Your included editable files
- `src/models/__init__.py`
- `src/models/baselines.py`
- `src/training/__init__.py`
- `src/training/optimization.py`
- `src/training/train.py`
- `tests/test_models/test_training.py`
- `tests/test_models/test_optimization.py`

Generated output and new documentation ownership is listed in the shared guide. `SOURCE_MANIFEST.csv` verifies these unchanged baseline copies; line references below refer to this original baseline and will move as code changes.

## Your objective
Produce a correct candidate scorer and reproducible five-model comparison, then a justified tuned artifact for the server. Own training methodology and model provenance. About ten relative effort units cover scoring/CV, experiments/tuning, artifacts and tests.

## What already exists
`baselines.py:24` defines LogisticRegression, RandomForest, HistGradientBoosting, KNN and MLP_NeuralNet. `train.py:23` performs GroupKFold; `train.py:83` trains/saves models and results; `optimization.py:32` runs Optuna TPE for RF. The archive already includes model weights, feature metadata and experiment tables. Reproduce evidence after fixing scoring; do not infer model quality from their presence.

## Implementation tasks, in order
1. Fix Top-1 selection in both `src/models/baselines.py:132` and `src/training/train.py:62`. Cumulative sums remain 1 for non-winning rows after a first maximum, so several rows may be flagged as Top-1. Select exactly one maximum per `(trip_id, gps_index)` with a deterministic tie rule, such as smallest arc ID. Share one implementation rather than duplicating broken logic. Preserve alignment even with shuffled/nonunique DataFrame indices.
2. Fix the scorer boundary at `baselines.py:120`: it currently retains only keys and optional label, dropping `dist_to_arc`, `cos_heading_diff` and `projection_fraction`. Preserve these geometry fields for C's decoder, preferably all original feature columns. Do not select them as extra classifier inputs outside the frozen `FEATURE_COLUMNS`. Validate missing columns, finite values, empty inputs, positive-class mapping and probability bounds.
3. Audit CV/refit: training trips only, disjoint trip IDs across folds, enough unique groups, reproducible seeds and preprocessing fitted inside each fold. Use cloned estimators where needed. MLP's internal early-stopping split is row-based; document or revise it if claiming every validation mechanism is trip-aware. Reject missing labels instead of casting NaN. Read trip IDs as strings to preserve leading zeroes.
4. Train all five existing families on A's documented data/sample and compare real CV Macro F1, student-test Macro F1, observation Top-1, log loss and inference time. Call C's corrected metrics; graph-free numeric-ID proximity is not road continuity. Where true graph information is unavailable, report N/A with explanation rather than a misleading score. Save configurations and confusion matrices as appropriate, and actual plots (including log-loss comparison where applicable).
5. Choose the model using a declared training-CV objective and operational constraints. Keep student-test as disclosed development evaluation. The optimizer is currently hard-coded to RF; justify tuning RF from the comparison, or implement Bayesian tuning for the selected family. Save search space, objective, seed, actual trial count, full trials, best parameters and tuned-versus-untuned comparisons on identical partitions. Twelve trials is the README example, not evidence that you ran twelve trials.
6. Persist the final model and metadata, including ordered feature names, versions, seed, training trip list/sample identity and schema/config identity. Coordinate artifact filename and model display name with D: the API currently tries tuned_model then RandomForest and hard-codes RF names. Restart the API after replacing artifacts. Rebuild result tables/plots when Top-1 or metric repairs change outputs.

## Handoff and dependencies
Use synthetic fixtures for scorer/CV unit work while waiting for A; use real training data for final results. Request A's small multi-trip sample early and freeze the schema. Request C's metric definitions early, then provide C/D the repaired scorer before full experiments finish. You own all `models/`, `results/optimization/` and model comparison outputs except C's `reconstruction_comparison.csv`.

Do not edit metrics or feature generation. Where a training call needs a new metric option, agree it with C and change only your consumer. Put model-specific explanations, actual hyperparameters, selection evidence, tuning tables and limitations in `docs/notes/member_b.md` for D's consolidated report.

## Verification and completion
Run `python -m pytest tests/test_models/test_training.py tests/test_models/test_optimization.py -v`. Add regression cases where the best candidate is first/middle/last, all probabilities tie, trip IDs repeat across batches, indices are shuffled, and decoder geometry survives scoring. Assert exactly one flag and the correct selected arc in each group. Test fold trip disjointness explicitly; the existing CV test only checks returned score bounds. The existing optimization test uses two synthetic trials and two folds; it is a smoke test, not final experiment evidence.

After A's caches are verified, existing entry points are `train_models(train_df, test_df)` and `optimize_random_forest(train_df, n_trials=12)`; use the latter only if RF is the chosen justified target and record actual arguments. Save commands and versions. You are done when scorer regressions pass, five models and tuning are reproducible, the final model loads with recorded dependencies, and C/D consume its scores/schema successfully. Deliver code, models, metadata, result tables/plots, trial history, test output and notes.
