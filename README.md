# Auditable Depression Graph Neural Network
Graph neural networks for detecting depression in clinical interview transcripts (DAIC-WOZ, E-DAIC), built on an inspectable PHQ-8/DSM-5 to Empath mapping with no LLM in feature extraction. Includes per-case audit reports of criteria, categories and matched terms. Research code, not a diagnostic tool.

## Master's dissertation repository

This is the code behind a master's dissertation in Computer Science at the Graduate Program in Informatics (PPGINF), Pontifícia Universidade Católica de Minas Gerais (PUC Minas):

> R. B. Sze, *Representations of Major Depressive Disorder from Clinical Interview Discourse*,
> master's dissertation, PUC Minas, 2026. Advisor: Prof. Wladmir Cardoso Brandão.

The work has two published or in-progress outputs, and this repository supports both:

| Output | Status | Role |
|---|---|---|
| **Survey**: Sze & Brandão, "A Comprehensive Survey on Datasets for Affective Computing and Mental Disorder", *IEEE Transactions on Affective Computing*, 2025, DOI [10.1109/TAFFC.2025.3624354](https://doi.org/10.1109/TAFFC.2025.3624354) | Published | Its dataset taxonomy was used to justify the choice of DAIC-WOZ for this work. |
| **Paper**: "Knowledge-guided Hybrid Graph Learning for Depression Detection" (Sze & Brandão) | **In development, not yet submitted or peer reviewed** | Presents the models and the auditability analysis in this repository. | 

Nothing here has been peer reviewed as a method. Numbers reported in the dissertation and in the draft may change before submission.

## Briefing

Recent systems often reach strong benchmark scores by using a large language model to extract psychological features from text. That swaps one black box for another and leaves a clinician unable to see why a case was flagged. This project takes the opposite design choice:

- The eight PHQ-8 criteria (derived from the DSM-5) are mapped by hand to Empath lexical categories, with an expected direction of change in depressed participants. The mapping is a plain JSON file: [`data/mapping/phq8_dsm5_empath.json`](data/mapping/phq8_dsm5_empath.json).
- Three graph architectures isolate each component: a lexical graph baseline (InductGCN), a graph over Empath categories (EmpathGCN), and a graph-attention model with temporal and paralinguistic features (GAT-GCN-FeExtr), the last trained as an ensemble of 20 models.
- Training uses an asymmetric loss and a threshold constrained by a maximum number of false positives, because a missed case costs more than a false alarm in screening.
- A per-case audit report lists the criteria, categories, matched terms, and sentences behind a result. See [`docs/AUDITABILITY.md`](docs/AUDITABILITY.md).

## Status and limitations
Read these before relying on anything here.

- **Development-set results are optimistic.** The class-weight multiplier, decision threshold, hyperparameters, and random seed were selected on the DAIC-WOZ development set, where results are also reported. The cross-corpus E-DAIC evaluation is the less biased evidence.
- **Small samples.** The development set has 35 sessions (12 depressed).
- **E-DAIC transcripts come from automatic speech recognition**, which lowers text quality and is one factor in the drop from DAIC-WOZ to E-DAIC.
- **Mapping coverage gap.** 14 categories named in the mapping do not exist in the standard Empath lexicon and silently score zero. `tests/test_mapping.py` documents this. An optional extended mode uses proposed term lists that have not been clinically reviewed.
- **Text only.** No audio or video signal is used.
- **No clinical validation.** No psychologist has evaluated the audit reports. The scores describe
  vocabulary, not a person's clinical state.
- Higher scores have been reported by other systems that rely on LLM-extracted features and
  synthetic training data. This project does not try to match them by that route.

## Data
DAIC-WOZ and E-DAIC are distributed by USC ICT under an End User License Agreement and are **not
included**. See [`DATA_ACCESS.md`](DATA_ACCESS.md). Anything under `data/processed/` and `results/`
is derived and does not allow reconstruction of transcripts.

## Repository layout

```
data/mapping/        PHQ-8/DSM-5 to Empath mapping (JSON)
src/common/          preprocessing, feature extraction, evaluation, auditability
src/paper1_graph_model/   graph construction and the three architectures
notebooks/           exploratory analysis and experiments
papers/paper1_taffc/ manuscript draft and reproducibility checklist
docs/                auditability notes
tests/               integrity checks for the mapping and report code
```

Several notebooks and modules are still placeholders while code is being ported from the original
Colab notebooks. [`papers/paper1_taffc/reproducibility_checklist.md`](papers/paper1_taffc/reproducibility_checklist.md)
tracks what is done.

## Citation

If you use this code, cite the dissertation, and the survey if you use its taxonomy:

```bibtex
@mastersthesis{sze2026representations,
  author = {Sze, Rodrigo Bessa},
  title  = {Representations of Major Depressive Disorder from Clinical Interview Discourse},
  school = {Pontif\'icia Universidade Cat\'olica de Minas Gerais},
  year   = {2026}
}

@article{sze2025survey,
  author  = {Sze, Rodrigo Bessa and Brand{\~a}o, Wladmir Cardoso},
  title   = {A Comprehensive Survey on Datasets for Affective Computing and Mental Disorder},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2025},
  doi     = {10.1109/TAFFC.2025.3624354}
}
```

A citation for the paper will be added once it is available. See also [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). The license covers code only, not
DAIC-WOZ or E-DAIC data.


Rodrigo Bessa Sze, PPGINF, PUC Minas. rodrigo.sze@gmail.com
