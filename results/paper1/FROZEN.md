# Frozen configuration for scoring the external E-DAIC cohorts

Recorded on 2026-11-25, before the external E-DAIC cohorts defined below were scored for the first
time. Nothing listed here is changed after scoring.

- Code commit: `0f439b014ff75ad460db8817d9b2f77940688074` (branch `fix/edaic-cohort`)
- Notebook: `notebooks/02_experiments.ipynb`

## Cohorts

| Name | Role | Rule |
|---|---|---|
| `daic_train` + `daic_dev` | training pool (140, 42 positive) | official AVEC 2017 splits, usable transcripts |
| `daic_test` | held-out (46, 14 positive) | official AVEC 2017 test split |
| `edaic` | held-out, external (86, 29 positive) | labeled E-DAIC participants (official train, dev and test splits) whose ID appears in no DAIC-WOZ split |
| `edaic_test` | held-out, external (56, 21 positive) | official E-DAIC test split (IDs 600-718), scored as its own cohort |
| `asr_daic_test` | controlled analysis only (46, 14 positive) | DAIC-WOZ test participants on their E-DAIC ASR transcript; written to `controlled_*` files, never part of the held-out evaluation |

Labels: PHQ-8 total >= 10. For E-DAIC the binary label is recomputed from the released total.
`src/paper1_graph_model/corpus.py` raises an error if `edaic` or `edaic_test` shares a participant
with any DAIC-WOZ split.

## Features

Temporal, paralinguistic and psycholinguistic views as in `src/common/features.py` at the commit
above, including the 19-word absolutist dictionary of Al-Mosaiwi and Johnstone (2018) and the
PHQ-8/DSM-5 to Empath mapping in `data/mapping/phq8_dsm5_empath.json`.

## Protocol

- Fixed structure: hidden 32 (64 for InductGCN), 2 heads, dropout 0.5, 250 lexical terms, window 20,
  150 Empath categories.
- Optuna study version `v2.1` (storage `OUTPUT_DIR/optuna.db`), reused as completed: 30 TPE trials
  per model, restricted space (lr, weight decay, epochs, m), objective = out-of-fold PR-AUC of 2x5
  repeated stratified CV on the pool. `N_FOLDS = 5`, `N_REPEATS = 2`, `SEED = 42`.
- Cross-fitted ensemble: 5 folds x 4 seeds = 20 members.
- Threshold rule: fewest false negatives with false-positive rate <= 6/21 on the ensemble's
  out-of-fold pool probabilities, tau in [0.20, 0.60]. The rule is frozen, not re-chosen.

## Selected hyperparameters (study v2.1, `results/experiments/v2/best_params.json`)

| Model | lr | weight decay | epochs | m | CV PR-AUC |
|---|---|---|---|---|---|
| InductGCN | 0.000406 | 0.000560 | 450 | 2.874 | 0.519 |
| GAT text-only | 0.000205 | 0.000103 | 300 | 4.599 | 0.474 |
| GCN+GAT hybrid | 0.000201 | 0.003815 | 450 | 4.843 | 0.493 |
| EmpathGCN | 0.005397 | 0.000163 | 400 | 2.469 | 0.638 |
| GAT-GCN-FeExtr (Empath graph) | 0.002412 | 0.000117 | 600 | 3.288 | 0.582 |
| GAT-GCN-FeExtr (word graph) | 0.000179 | 0.000982 | 350 | 3.019 | 0.498 |

## Thresholds of the previous run (reference)

The ensembles are retrained when the notebook runs, so tau is recomputed by the frozen rule from
the same pool and hyperparameters. Models over the Empath graph are not bit-reproducible on GPU;
any change of tau from these values is reported, not corrected.

| Model | tau | pool FP (limit 28) |
|---|---|---|
| InductGCN | 0.60 | 30 |
| GAT text-only | 0.60 | 30 |
| GCN+GAT hybrid | 0.60 | 34 |
| EmpathGCN | 0.42 | 28 |
| GAT-GCN-FeExtr (Empath graph) | 0.53 | 23 |
| GAT-GCN-FeExtr (word graph) | 0.60 | 28 |

Main model, fixed before this run: GAT-GCN-FeExtr (Empath graph).
