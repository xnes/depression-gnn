# Auditable Depression Graph Neural Network

Graph neural networks for detecting depression from clinical interview transcripts (DAIC-WOZ,
E-DAIC), built only on inspectable sources: lexical statistics, timestamps, transcribed
paralinguistic tags, and an explicit mapping of the PHQ-8 criteria (DSM-5) to Empath lexical
categories. No large language model is used anywhere in feature extraction. Research code, not a
diagnostic tool.

This repository accompanies the master's dissertation

> R. B. Sze, *Representations of Major Depressive Disorder from Clinical Interview Discourse*,
> master's dissertation, PPGINF, PUC Minas, 2026. Advisor: Prof. Wladmir Cardoso Brandão.

and the paper "Knowledge-guided Hybrid Graph Learning for Depression Detection" (Sze and Brandão),
which is in preparation and has not been peer reviewed.

## Method in brief

- **Mapping.** [`data/mapping/phq8_dsm5_empath.json`](data/mapping/phq8_dsm5_empath.json) links
  each PHQ-8 criterion to Empath categories with an expected direction of change
  ([description](data/mapping/phq8_dsm5_empath.md)). It yields one score per criterion, used to read
  the evidence behind each prediction. The graph itself is data-driven.
- **Graphs.** A word graph (TF-IDF, windowed PPMI) and a graph of Empath categories selected by
  ANOVA on the training data. New participants are added inductively.
- **Models.** InductGCN (lexical baseline), EmpathGCN, and GAT-GCN-FeExtr, a graph-attention model
  that also receives temporal, paralinguistic, and criterion-score features.
- **Protocol.** DAIC-WOZ train and dev form a training pool of 140 participants. Hyperparameters and
  the decision threshold are chosen by cross-validation inside the pool. A cross-fitted ensemble of
  20 models is scored once on the DAIC-WOZ test partition (46) and on E-DAIC (112, excluding
  sessions shared with DAIC-WOZ training).
- Only participant speech is used; interviewer turns contribute timestamps only.

## Limitations

- Small evaluation cohorts; confidence intervals are wide and model differences are not significant.
- E-DAIC transcripts come from automatic speech recognition and carry no paralinguistic tags.
- 14 categories named in the mapping do not exist in Empath 0.89; they are excluded from criterion
  scores.
- Labels are PHQ-8 self-reports (a screening instrument), not clinical diagnoses. No clinician has
  evaluated the mapping or the audit reports.

## Data

DAIC-WOZ and E-DAIC are distributed by USC ICT under an End User License Agreement that prohibits
redistribution, and are **not included**. Request access from USC ICT
(https://dcapswoz.ict.usc.edu/), then point the environment variable `DAIC_DATA_DIR` to the
corpora, with the layout described in [`src/common/data.py`](src/common/data.py).

Everything under `results/` is aggregate (counts, statistics, metrics, figures) and does not allow
the reconstruction of transcripts or the identification of participants.

## Reproducing

Both notebooks run in Google Colab with Google Drive mounted; edit the paths in their setup cell.

1. [`notebooks/01_dataset_eda.ipynb`](notebooks/01_dataset_eda.ipynb): cohort characterization and
   exploratory analyses, written to `results/eda/`.
2. [`notebooks/02_experiments.ipynb`](notebooks/02_experiments.ipynb) (GPU recommended): tuning,
   ensembles, and evaluation, written to `results/experiments/v2/`. Per-participant probabilities
   and attention go to an output directory outside the repository.
3. Paired comparisons and auditability analyses, from that output directory:

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/paired_auc.py <OUTPUT_DIR> results/experiments/v2
PYTHONPATH=. python scripts/auditability.py <OUTPUT_DIR> results/experiments/v2
PYTHONPATH=. python -m pytest -q tests
```

## Layout

```
data/mapping/            PHQ-8/DSM-5 to Empath mapping and its description
src/common/              data loading, features, evaluation, exploratory analysis
src/paper1_graph_model/  graph construction, models, experiment protocol
scripts/                 paired bootstrap and auditability analyses
notebooks/               01 exploration, 02 experiments (with outputs)
results/                 aggregate results and figures
tests/                   unit tests
```

## Citation

```bibtex
@mastersthesis{sze2026representations,
  author = {Sze, Rodrigo Bessa},
  title  = {Representations of Major Depressive Disorder from Clinical Interview Discourse},
  school = {Pontif\'icia Universidade Cat\'olica de Minas Gerais},
  year   = {2026}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

MIT ([`LICENSE`](LICENSE)). The license covers the code only, not DAIC-WOZ or E-DAIC data.
