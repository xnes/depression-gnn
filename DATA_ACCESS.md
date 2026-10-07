# Data Access

This repository does **not** include the DAIC-WOZ or E-DAIC (AVEC 2019) raw audio,
video, or transcripts. Both corpora are distributed by the University of Southern
California Institute for Creative Technologies (USC ICT) under an End User License
Agreement (EULA) that prohibits redistribution.

## Requesting access

1. Visit the DAIC-WOZ / AVEC corpus request page maintained by USC ICT.
2. Complete and sign the EULA as an individual researcher or on behalf of your
   institution.
3. Once access is granted, place the raw files under `data/raw/` using the following
   expected layout (not tracked by git — see `.gitignore`):

```
data/raw/
├── daic_woz/
│   ├── <participant_id>_TRANSCRIPT.csv
│   └── ...
└── e_daic/
    ├── <participant_id>_Transcript.csv
    └── ...
```

## What we DO distribute

- All code under `src/` and `notebooks/` (MIT licensed).
- Aggregate, non-reidentifiable derivatives under `data/processed/` and `results/`
  — e.g., per-participant feature vectors (Empath category scores, timing/linguistic
  feature summaries), model checkpoints' evaluation metrics, and figures. None of
  these allow reconstruction of the original transcript text.

## A note on E-DAIC transcript quality

E-DAIC transcripts were produced via Automatic Speech Recognition (ASR), unlike
DAIC-WOZ, whose transcripts were manually produced and reviewed. This introduces a
higher error rate in E-DAIC text and is a documented contributor to the performance
drop observed when models trained on DAIC-WOZ are evaluated on E-DAIC (see
`notebooks/paper1_graph_model/07_cross_validation_e_daic.ipynb` for the diagnostic
analysis).
