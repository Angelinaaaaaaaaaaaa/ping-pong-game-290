# Human behavior versus V3 robotic skill selection

This package analyzes saved human reconstructions and robot decision logs.
Human categories are inferred realized landing regions; robot categories are
logged selected commands. Similarity does not establish identical intentions,
realized robot placement, or playing strength. No analysis command trains a
model or runs a robot evaluation.

Install the analysis dependencies (Python 3.10+):

```bash
python3 -m pip install -r v3/human_comparison/requirements.txt
```

Run commands from the repository root. Dataset archives, reconstruction arrays,
checkpoints, evaluation CSVs and generated reports are local inputs/outputs;
they are not included with the comparison code. Existing robot logs used here
are `skill_eval/v3/nash_vs_random_10000/decisions.csv` and
`skill_eval/v3/potential_ibr_vs_random_10000/decisions.csv`. Each should retain
its `summary.json` metadata. CLI options can select alternative paths.

## Reproduce the saved-data comparison

If the event-level human CSVs already exist in
`v3/human_comparison/reports/10k_both/`, start here. No LATTE-MV re-extraction is
needed. The folder must contain `shot_events.csv`, `all_shot_events.csv`, and
`summary.json`. Keep the source files unchanged.

First create the spatial reference/configuration from saved events:

```bash
python3 v3/human_comparison/analyze_spatial_v2.py \
  --human-dir v3/human_comparison/reports/10k_both \
  --primary-method four_quadrants_center_override \
  --output-dir v3/human_comparison/reports/spatial_comparison_v3_four_quadrants_center_override
```

Then analyze both human players and the policy-controlled P1 robot decisions:

```bash
python3 v3/human_comparison/analyze_opponent_position.py \
  --human-events-dir v3/human_comparison/reports/10k_both \
  --human-players both \
  --output-dir v3/human_comparison/reports/opponent_position_comparison_10k_both

python3 v3/human_comparison/analyze_joint_position.py \
  --input-dir v3/human_comparison/reports/opponent_position_comparison_10k_both \
  --output-dir v3/human_comparison/reports/joint_position_comparison_10k_both

python3 v3/human_comparison/plot_joint_landing_maps.py
```

Add the empirical Random P2 observations from each robot run and the exact
uniform action-distribution reference:

```bash
OPENBLAS_NUM_THREADS=1 python3 v3/human_comparison/analyze_random_similarity.py

OPENBLAS_NUM_THREADS=1 python3 v3/human_comparison/analyze_center_band_sensitivity.py \
  --bands 0.05 0.15 0.25 --primary-band 0.15 --bootstrap 2000 --seed 42
```

For one selected width, use `analyze_random_similarity.py --center-band 0.25`.
It chooses a separate output directory by default. All commands expose `--help`.
Set `MPLCONFIGDIR=/tmp/human_comparison_mpl` if the default Matplotlib cache
directory is unavailable.

### Definitions and interpretation

- The primary landing method is `four_quadrants_center_override`: depth below
  0.51 is short and depth at/above 0.51 is deep; lateral at/above zero is the
  hitter's left and below zero is their right. The inclusive Center Safe
  rectangle, depth `[0.20, 0.70]` and lateral `[-0.35, 0.35]`, overrides all
  four base blocks. All valid table landings receive exactly one of five labels.
- Landing coordinates are normalized: depth 0 = net, 1 = opponent endline;
  lateral +1 = hitter's left, -1 = hitter's right. P2 landings are transformed
  once into that hitter-relative frame.
- Position Left is each player's own left: P1 +world-y and P2 -world-y.
  Physical pair order is always `(P1 position, P2 position)`. Role-relative
  sensitivity uses `(hitter position, defender position)` instead.
- Incoming contexts are recovered from the immediately preceding validated
  opponent shot. Missing incoming contexts are excluded; outgoing and
  pre-contact positions remain separate descriptive baselines.
- Position-center half-bands use `band_m / 0.7625` in normalized table units
  for both humans and robots. The shared pose-quality bound is retained.
  Wider bins redistribute existing observations; they do not add shots.
  Position Center and landing Center Safe are independent definitions.
- Raw category counts are pooled before normalization. No unweighted average
  of conditional vectors is used. Per-player counts, context frequencies,
  missing data and sample sizes are retained.
- Metrics include base-2 JS divergence (not square-root distance), TV distance,
  Spearman and Pearson correlation. Exact-uniform correlations are undefined.
  Empirical Random correlations can be unstable near a uniform distribution.
- Uncertainty resamples human matches and robot rallies. Both human players
  share their match cluster; learned and Random players in a robot run share
  episode weights. Different runs stay separate.
- Only recorded robot skills are used in the action comparison. No nominal
  target, planner prediction or action name substitutes for a measured bounce.
  Physical-pair comparisons mix attacker roles when humans are pooled and
  Random is P2; inspect opponent-only and role-relative results as well.

Each output directory contains CSVs, figures, a methodology report and source
hashes/configuration. `*_metrics_wide.csv` puts models side by side;
`all_similarity_metrics.csv` retains intervals and counts. Center sensitivity
also writes `primary_*_metrics_wide.csv`, `all_bands_*` CSVs, category-mix
tables, and nine-context landing scatter/density maps. The nearest-target,
balanced, compact and broad definitions remain separate sensitivity methods.

## Optional small-sample event extraction

The extraction command below is for preparing a new pilot when saved human
events are unavailable. Its default labels are the historical provisional
nearest-target baseline; the saved-data comparison above explicitly replaces
them with the four-block classifier.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/latte_pilot_matplotlib python3 v3/human_comparison/analyze_human.py \
  --data-root v3/LATTE-MV --sample-rallies 20 --seed 42 --player p1 --verbose
```

The reader accepts a ZIP path, a directory containing `release_data.zip`, or
an extracted `release_data` tree. It reads only sampled arrays/auxiliary files
and at most three small video clips into temporary storage. It never extracts
the entire archive or modifies the dataset. ZIP filenames are indexed once;
rally arrays are processed sequentially. No network is needed to run it.

Default output: `v3/human_comparison/reports/`. Start with
`dataset_investigation.md` and `pilot_report.md`. The sample manifest records
exact reconstruction paths and SHA-256 hashes. Use `--sample-manifest` to
replay its exact member list; `--seed` sampling uses match-balanced selection.
`--player p2` and `--player both` use the same oriented table geometry.
`--position-time pre_contact`, `--position-landmark pelvis`,
`--center-normalized`, `--edge-margin-m`, and `--net-clearance-m` support
sensitivity checks. Keep alternative runs in separate `--output-dir` folders.

Human labels are **provisional nearest-target landing proxies**, derived from
V3 target points via `--robot-profile`. They are not commanded skills or
ground-truth human annotations. Original normalized landings remain available.
Contact/bounce anchors and interpolated trajectories are partly circular
validation evidence; inferred bounces may be clipped by the upstream code.
Robot incoming-crossing conditions and human outgoing-crossing conditions
also differ temporally. Resolve these points before scaling or interpreting
human/robot similarity.

Outputs:

- `all_shot_events.csv`: both players, including exclusions and reasons.
- `shot_events.csv`: requested players, including excluded shots.
- `joint_position_skill_summary.csv`, `opponent_position_skill_summary.csv`:
  reuse the robot CSV schema; `attack_skill` keys denote provisional proxies.
- `position_combination_frequency.csv`: all nine cells and occupancy counts.
- `summary.json`: counts, five-dimensional empirical vectors, configuration,
  video checks and robot comparison sources. Empty vectors are null.
- `sample_manifest.json`: sample IDs, input hashes, dimensions and quality.
- Conditional grids, top-down trajectories, height/time traces, positioning
  zones, provisional target regions, original-video frame sheets, side-by-side
  human/Nash-Q/Potential IBR compatibility figure.

There are no logged human policy vectors. Thus `n_policy_vectors=0` and
`mean_policy_probability` is blank. `empirical_probability` is a count ratio,
and empty conditions are undefined. The same robot summary/plot helpers are
imported without modification. With `--robot-summary`, explicit robot summary
CSV paths can replace automatic discovery of existing reports.

Dependencies are NumPy, SciPy, pandas, matplotlib and existing V3
analysis imports. OpenCV is used only for the optional video frame sheets;
disable these with `--video-examples 0` if unavailable.

```bash
MPLCONFIGDIR=/tmp/latte_pilot_matplotlib python3 -m unittest discover \
  -s tests -p test_v3_human_comparison.py
```

## Potential Max and the latest human-only/side-by-side bar charts

For a detailed Potential Max P1 versus Random P2 log:

```bash
OPENBLAS_NUM_THREADS=1 python3 v3/human_comparison/analyze_random_similarity.py \
  --potential-policy potential_max \
  --potential-decisions skill_eval/v3/potential_max_vs_random/decisions.csv \
  --center-band 0.25 --bootstrap 2000 --seed 42 \
  --output-dir v3/human_comparison/reports/potential_max_vs_human_center_0p25

python3 v3/human_comparison/plot_human_rectangular_skill_bars.py
python3 v3/human_comparison/plot_human_robot_skill_bars.py
```

The first plotting script generates human-only bars for all nine physical
pairs, recomputing the four-block + Center Safe labels from coordinates.
The second aligns those human bars with Nash-Q and Potential Max selected
skills, both in a grouped 3x3 chart and nine individual three-panel figures.
The latest graphs explicitly name the rectangular classifier; the original
nearest-target extraction figures remain preserved as a separate baseline.

Detailed Potential Max runtime and evaluator documentation is in
[../POTENTIAL_MAX.md](../POTENTIAL_MAX.md).
