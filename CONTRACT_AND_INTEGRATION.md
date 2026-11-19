# Four-person ownership and integration guide

## Basis and limits
This is a work allocation based on the actual uploaded `PathSnap_Phase1_Team_Scaffold.zip`, inspected on 18 November 2026. Source ZIP SHA-256: `18990cbb9d4adfd90396afe32190deb342e51a7ffa7a38eac67b0eeda4be67c5`.

The archive already contains implementations, five baseline model artifacts, a tuned model, compressed feature tables, plots, an API, a Streamlit UI and a technical report. Work here means reviewing, correcting, completing and reproducing those components, not pretending they are empty stubs. The old `member_assignments/` and `docs/INTEGRATION_CONTRACT.md` describe three members; this four-person plan supersedes that ownership division. Source behavior takes precedence over their outdated example signatures.

Static inspection verified 80 `train` trip folders and 10 `student_test` folders, each with `.track`, `.nodes`, `.arcs` and `.route` files. No extracted `evaluator_test` directory is present. Student-test IDs are `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`.

Sample raw files have whitespace-separated track rows `(lon, lat, time)`, node rows `(lon, lat)`, arc rows `(u, v)`, and one route arc ID per row. Loader IDs are zero-based row positions. Code inspection found 26 test functions; this handoff did not execute them, retrain models, load serialized models, or independently validate saved accuracy claims. Existing README/results are historical evidence to reproduce, not newly certified results.

## Using each download
Each ZIP contains `START_HERE.md`, this guide, `owned/` with unchanged original assigned files, `SOURCE_MANIFEST.csv` with their hashes, and `reference/` with original context. It is an implementation handoff, not a newly implemented or standalone application.

Everyone starts from the same full original scaffold, including `data/`, `models/` and all modules. Extract your handoff separately, read it, and work in that full project checkout. `owned/` paths are relative to the project root: they are the files you may edit. They are already present in the original scaffold; there is no need to overwrite a teammate's newer work. Do not copy the handoff directory itself into Python's import path. Large datasets, models, results, `.git`, caches, `.coverage` and bytecode were deliberately not duplicated into the four downloads. Keep the original ZIP available.

## Exclusive ownership
| Member | Editable implementation and tests | Generated outputs / documentation |
|---|---|---|
| A | `src/data/`, `src/features/`, `tests/test_data/`, `tests/test_features/` | `data/processed/`, `results/eda/`, `docs/notes/member_a.md` |
| B | `src/models/`, `src/training/`, `tests/test_models/test_training.py`, `tests/test_models/test_optimization.py` | `models/`, `results/optimization/`, model-comparison files except C's reconstruction comparison, `docs/notes/member_b.md` |
| C | `src/reconstruction/`, `src/evaluation/`, `tests/test_reconstruction/`, `tests/test_models/test_metrics.py`, `tests/test_integration_pipeline.py` | `results/maps/`, `results/model_comparison/reconstruction_comparison.csv`, `docs/notes/member_c.md` |
| D | `src/api/`, `frontend/`, `tests/test_api/`, `run_pathsnap.py`, `run_pathsnap.bat`, `requirements.txt`, `.gitignore` | Root README, project folder README files, `docs/INTEGRATION_CONTRACT.md`, `docs/PHASE1_CHECKLIST.md`, consolidated technical report, `member_assignments/`, `results/logs/`, `docs/notes/member_d.md` |

Empty `src/__init__.py` stays unchanged unless D coordinates a packaging change. Original assessment/guideline documents and raw inputs remain reference material. Existing artifacts are not new contributions. Each person writes their own method/results notes; D assembles them into one team report. New files belong under the owner's listed directories; propose exceptions to that owner before editing.

The split balances complexity, not file count. Budget about ten relative effort units each (planning estimates, not measured hours): A parsing/EDA 3 + candidate geometry 3 + labels/features 2 + tests/handoff 2; B scoring/CV 3 + experiments/tuning 3 + artifacts 2 + tests/handoff 2; C decoder/bridging 3 + metrics/comparisons 3 + integration 2 + tests/handoff 2; D API contracts 3 + UI/remote behavior 3 + deployment/report assembly 2 + tests/handoff 2. B's runs can take substantial machine time; C's fewer files contain the algorithm-heavy work. D compiles report contributions rather than writing everyone's analysis.

## Contracts to preserve and explicitly finish
1. **A → B/C/D:** `load_trip(trip_id, partition='train', data_root=...)` returns `trip_id`, `partition`, `track`, `nodes`, `arcs`, `route`. Track has `gps_index`, `lon`, `lat`, `time`, `dt`, `speed_m_s`, `bearing_deg`; arcs have `arc_id`, `u`, `v`, `length_m`, `bearing_deg`. Distances in metres, time in seconds, bearings in degrees. Arc/node IDs are trip-local, not global.
2. **A → B:** `build_training_dataset(trip_ids=None, data_root=..., partition='train', max_trips=None)` returns candidate rows. Composite row key is `(trip_id, gps_index, candidate_arc_id)`; preserve trip IDs as strings when reading caches: `pd.read_csv(path, dtype={'trip_id': str})`. Label is 0/1 only when permitted ground truth exists; absent truth is missing, never silently negative.
3. **A → all:** `FEATURE_COLUMNS` in `src/features/dataset.py` is the ordered source of truth: `dist_to_arc`, `projection_fraction`, `dist_to_start_node`, `dist_to_end_node`, `arc_length`, `heading_diff_deg`, `cos_heading_diff`, `sin_heading_diff`, `gps_speed`, `gps_delta_t`, `dist_rank`, `dist_ratio_to_nearest`, `normalized_dist`, `speed_length_ratio`. No label, trip ID, candidate ID or reference route enters X. Any feature change requires cache regeneration, retraining and metadata updates before integration.
4. **B → C/D:** `predict_candidates(feature_dataframe, model, feature_cols=None)` must preserve keys plus decoder geometry fields `dist_to_arc`, `cos_heading_diff`, `projection_fraction`, and add finite `probability`, `score`, `top1_predicted`. Preserve an optional label only for evaluation. Exactly one row per `(trip_id, gps_index)` has Top-1 flag 1, with deterministic ties. This is a required repair: current scoring drops geometry and has incorrect tie logic.
5. **B → D:** supply `models/tuned_model.joblib` and `models/feature_metadata.json`, with feature order, library versions, seed, training trip list/cache identity and selection rationale. Current API also accepts `models/RandomForest.joblib` as a fallback; any change to selected model filenames or displayed model identity is agreed with D. Optimization currently targets RF; if another family wins, B must justify RF tuning or implement Bayesian tuning for the chosen family rather than claim RF is automatically best.
6. **C → D:** actual signature is `decode_route(track, candidate_predictions, road_graph, arcs=None, config=None)`; always pass `arcs=arcs`. It returns `pointwise_arcs`, `full_route_arcs`, `log_likelihood`. Pointwise output aligns one-for-one with GPS rows; full route must include connecting arcs when reachable. A disconnected gap must be explicitly reported under an agreed extension or rejected clearly, not silently described as continuous. D records any added response fields before using them in the UI.
7. **C → B/D:** keep `evaluate_candidate_classification`, `compute_top1_accuracy`, `compute_route_continuity`, `evaluate_model`. Use true topology for continuity, not proximity of numeric arc IDs. Agree on handling missing graph data (N/A or an explicit error), singleton trips and zero eligible transitions. Keep candidate binary Macro F1, observation Top-1 accuracy, pointwise adjacency and expanded-route continuity distinct. A's closest-reference-arc mapping gives derived observation labels, not directly supplied timestamped labels.
8. **D → UI:** current routes are `GET /health`, `GET /trips?partition=...`, `POST /predict/{trip_id}?partition=...`; POST has no required JSON body. Prediction keys include `trip_id`, `partition`, `status`, `model_used`, `inference_duration_seconds`, `n_observations`, `n_reconstructed_arcs`, `metrics`, `gps_points`, `full_route_arcs`, `route_segments_coords`, `ground_truth_coords`. Metrics currently include `observation_accuracy`, `route_continuity`, `duration_minutes`. GPS objects use named `lat`/`lon`; map segment arrays use `[lat, lon]`. Missing permitted labels means null accuracy and no truth overlay.

## Dependency sequence and working in parallel
**Gate 0 — start together:** D creates the common base and four branches, records this ownership map in the repository, and updates the old contract after all four review signatures. A confirms units/label semantics; B/C agree on the scorer columns and Top-1 rule; C/D agree on gap reporting and metric definitions. No silent interface changes.

**Gate 1 — independent development:** A validates one real training trip and one student-test trip, then exports a small labelled training fixture from multiple training trips. B can start with synthetic unit fixtures, but final model evidence uses real data. C builds tiny hand-checkable road graphs and tests decoder edge cases. D can test response schemas with clearly labelled fixtures while building remote-only UI behavior. Fixtures never count as final performance evidence.

**Gate 2 — data/model boundary:** merge A's validated data pipeline; B integrates actual feature rows, fixes scorer behavior, runs group-aware CV and trains models. Merge B's schema/scorer repair early so C and D can consume correct geometry while experiments continue. C delivers corrected metrics early; B reruns comparison outputs using them.

**Gate 3 — route/application boundary:** merge C's decoder and topology metrics, then D's orchestration/UI updates. B refreshes result tables affected by scoring/metric changes. C runs the full data → scoring → decoding test. D tests actual API JSON and client rendering.

**Gate 4 — release:** finish B's model/results update, clear D's model cache by restarting the API, and run the complete suite against the combined tree. D runs the two-laptop demo; all members supply notes and review their report sections. Do not merge stale model/cache artifacts after a feature-schema change. Any fix that changes predictions invalidates affected saved metrics and plots.

## Setup and merge procedure
Run commands from the full project root, preferably with Python 3.11 as advertised by the scaffold. D must verify and record the actual compatible dependency versions; requirements are currently unpinned.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -v
```

Use branches `member-a/data-features`, `member-b/models-training`, `member-c/reconstruction-evaluation`, `member-d/api-ui-integration`. Create each from the same agreed integration base with `git switch -c <branch>`. Commit only owned paths; do not use blanket staging. Before proposing a merge, sync with the agreed integration branch, run your scoped tests, and attach changed-file list, exact commands, output paths, dependency/schema changes and known failures. The receiving owner reviews cross-boundary assumptions. D merges reviewed changes in the gate order above and reruns impacted checks; each file's owner resolves conflicts in that file. If sharing ZIPs instead of Git, send only changed owned relative paths plus the same evidence, and have D apply them to one integration copy.

## Final integration acceptance
- All module checks plus `python -m pytest tests -v` pass with saved output; investigate failures rather than replacing assertions with trivial bounds.
- Test a real student-test trip such as `00000052` and at least one larger trip. Exactly one Top-1 per observation; scorer geometry reaches decoding; arc IDs exist; expanded routes are directionally connected or gaps are explicitly represented.
- Verify evaluator handling using synthetic files in a temporary test directory, since no evaluator partition is shipped: never read evaluator `.route`, include labels in features, fit on those trips, or return evaluator truth. Student-test is development evaluation, not training data for CV/refit.
- On Laptop 1: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`. On Laptop 2: `python -m streamlit run frontend/app.py --server.port 8501`. Both use the same local network; set UI backend URL to Laptop 1's actual IP. Localhost only works on the same machine.
- Laptop 2 must obtain trip options and predictions from the remote API without local models/raw data. Stop the backend and confirm the UI shows a connection failure instead of running a hidden local inference fallback. Save successful receiving-server responses and UI/map evidence.
- README, model metadata, result tables and team report agree with the merged code. Report actual measured results and limitations, regenerate affected plots, and replace stale local `file:///c:/Users/Admin/...` links. The supplied submission guide calls for one consolidated report and repository; these four member packs are internal work handoffs.
