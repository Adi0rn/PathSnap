# Member A — Data, candidates and feature engineering

Read this file first, then `CONTRACT_AND_INTEGRATION.md`. Your archive contains original assigned source/test files under `owned/`; implementation tasks below remain to be completed. Work against the full original scaffold, not this partial archive alone.

Branch: `member-a/data-features`

## Your included editable files
- `src/data/__init__.py`
- `src/data/exploration.py`
- `src/data/loader.py`
- `src/features/__init__.py`
- `src/features/candidates.py`
- `src/features/dataset.py`
- `src/features/engineering.py`
- `tests/test_data/test_exploration.py`
- `tests/test_data/test_loader.py`
- `tests/test_features/test_candidates.py`
- `tests/test_features/test_engineering.py`

Generated output and new documentation ownership is listed in the shared guide. `SOURCE_MANIFEST.csv` verifies these unchanged baseline copies; line references below refer to this original baseline and will move as code changes.

## Your objective
Make raw trips become reproducible, correctly labelled candidate feature rows. Own data correctness and the feature schema, while B owns model behavior. You have about ten relative effort units, concentrated on geospatial and labelling correctness.

## What already exists
`src/data/loader.py:47` parses trips and derives speeds/bearings, `exploration.py:154` runs EDA, `candidates.py:46` uses STRtree with fallback, `engineering.py:15` computes features, and `dataset.py:83` builds labelled trip rows. Cached training/sample and student-test feature CSVs already exist in the original ZIP. They must be checked or regenerated after changes.

## Implementation tasks, in order
1. Inventory the actual partition IDs/files and record missing values, empty/malformed files, timestamp order/duplicates, coordinate ranges, invalid node/arc references and route availability. Validate bad inputs with clear errors; preserve trip-local row IDs. Verify lon/lat, metres and seconds against real samples. Add focused loader tests for the failure cases you support.
2. Correct and verify candidate search geometry. In `src/features/candidates.py:83`, `radius_m / max(lon_scale, lat_scale)` makes a circular degree buffer that is too narrow east-west at these latitudes. Use a suitable metric projection or conservative search followed by metric distance filtering. Handle an empty road network and enforce configured minimum/maximum counts: the current minimum fallback adds only one nearest arc. Define deterministic candidate ordering and valid configuration bounds.
3. Measure two separate quantities: fraction of observations with at least one candidate, and permitted-label recall of the true/derived reference candidate. Nearest fallback guarantees neither true-arc recall nor zero candidate misses in the semantic sense used by the README. Report results at several reasonable radii/caps and record fallback frequency and candidate counts.
4. Review `dataset.py:36`: nearest unique reference-route arc ignores route order and uses degree geometry. Document derived-label ambiguity for loops/parallel/repeated arcs; improve metric consistency and justify the adopted labelling rule. Reference labels may be used to build permitted targets and evaluate recall, never to select prediction candidates or features. Prove features are unchanged when reference labels are withheld.
5. Validate the 14 ordered features, zero-length arcs, initial heading, NaN/infinite values and row-key uniqueness. `get_feature_matrix` currently casts a present label column directly to integers, even if it contains NaN. Define training validation and inference behavior with B, keeping unlabelled rows out of supervised fitting.
6. Export reproducible training and student-test feature caches separately, with commands, selected trip list, sampling method/seed, row counts, class balance, candidate settings, schema order and input identity. Avoid splitting rows from one trip across train/validation. Produce EDA tables, trajectories, road-network view and a labelled Folium map under `results/eda/`.

## Handoff and dependencies
Start immediately from the raw data; you do not depend on model training. Give B an early multi-trip training sample and separate development rows, and give C/D one real loaded trip plus a feature schema example. Five-fold CV needs at least five distinct training trips. Freeze `FEATURE_COLUMNS` with B before long runs. If you change radius/labels/features later, notify B to rebuild artifacts and C/D to rerun integration.

Do not modify `src/models/`, metrics, API or UI. Ask those owners to adapt consumers. Your report contribution goes in `docs/notes/member_a.md`: dataset inventory, cleaning, feature rationale, derived-label limitations, measured candidate recall and EDA figure paths.

## Verification and completion
Run `python -m pytest tests/test_data tests/test_features -v`. Extend these owned tests with metric-radius, candidate-limit, malformed-input, no-label-leakage and empty/unlabelled dataset cases. Use real trip `00000001` from train and `00000052` from student_test for a smoke check; keep synthetic failure fixtures in temporary directories.

You are done when every valid observation has an explicitly defined candidate policy, all feature keys/units are documented, caches can be recreated, leakage checks pass, EDA is based on actual trips, and B/C/D can use your outputs without editing their interfaces. Deliver code, cache-generation command, data manifest, feature schema, EDA artifacts, test output and your notes. Do not repackage raw data or claim old cached results as new experiments.
