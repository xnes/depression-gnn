# PHQ-8 / DSM-5 to Empath mapping

This note documents [`phq8_dsm5_empath.json`](phq8_dsm5_empath.json), so that a reviewer can check
where each part of the mapping comes from, what supports it, and what it does not claim. The mapping
was built by hand and has not been reviewed by a clinician.

## 1. What the mapping is

Each of the eight PHQ-8 criteria is linked to a set of Empath lexical categories. Each category
carries an expected direction of change in depressed participants:

- **present**: the category is expected to be used more by depressed participants (positive
  correlation with the PHQ-8 score).
- **absent**: the category is expected to be used less (negative correlation).

The direction is a hypothesis to be checked against data, not a measured result.

## 2. The mapping

Categories in *italics* do not exist in the standard Empath lexicon (see Section 5).

| ID | DSM-5 criterion | PHQ-8 item | Present (expected increase) | Absent (expected decrease) |
|---|---|---|---|---|
| C1 | Depressed mood | `PHQ8_Depressed` | sadness, negative_emotion, suffering, pain, torment, *grief*, fear, nervousness, anger | — |
| C2 | Anhedonia | `PHQ8_NoInterest` | — | joy, positive_emotion, fun, *recreation*, play, love, affection, celebration |
| C3 | Appetite change | `PHQ8_Appetite` | eating, *food*, *hunger*, disgust, body | — |
| C4 | Sleep disturbance | `PHQ8_Sleep` | sleep, *fatigue*, health | — |
| C5 | Fatigue or loss of energy | `PHQ8_Tired` | *fatigue*, weakness, health, *tiredness* | — |
| C6 | Worthlessness or guilt | `PHQ8_Failure` | shame, negative_emotion, *failure*, weakness | achievement, *success*, *confidence* |
| C7 | Diminished concentration | `PHQ8_Concentrating` | confusion, *cognitive_dissonance* | *cognitive_processes*, *thinking*, reading |
| C8 | Psychomotor retardation or agitation | `PHQ8_Moving` | *motion*, movement, *speed* | — |

DSM-5 criterion A9 (thoughts of death or suicide) is not mapped, because the PHQ-8 omits that item.

## 3. Theoretical basis

The mapping rests on three layers of prior work. Each layer supports one link in the chain from
clinical criterion to word count.

| Layer | Reference | What it supports | What it does not support |
|---|---|---|---|
| 1. Clinical criteria | American Psychiatric Association, 2013 [`apa2013dsm`] | The eight rows of the table: each row is one DSM-5 symptom criterion for major depressive disorder, and the PHQ-8 item that measures it. | Which words or categories express a criterion. DSM-5 defines symptoms, not language. |
| 2. Lexicon validity | Fast, Chen and Bernstein, 2016 [`fast2016empath`] | That Empath categories are meaningful semantic groupings across affective, somatic and cognitive domains. The categories were generated from word embeddings, filtered by crowd workers, and validated against LIWC categories. | That a given Empath category indicates a given clinical symptom. That link is the author's choice. |
| 3. Lexical-clinical tradition | Pennebaker, Boyd, Jordan and Blackburn, 2015 [`pennebaker2015liwc`] | The general premise that counts of words in lexical categories capture measurable psychological states. | Empath itself: Empath is not LIWC, and LIWC's psychometric evidence does not transfer to Empath directly. |

The step from a DSM-5 criterion to specific Empath categories, and the expected direction, is not
taken from any of these references. It was made by the author and is the part a reviewer should
examine most closely.

## 4. Provenance

- Transcribed without change from the dictionary `DSM5_MAP` in `legacy/X-Empath 2.ipynb`, cell 7,
  lines 8–17.
- Added during transcription:
  - the `phq8_item` field, using the column names in `legacy/plotar.py`, lines 9–10;
  - the English `label` field;
  - the `missing_from_empath` lists.
- Empath version checked: 0.89, which has 194 categories.

## 5. Known limitations

1. **Fourteen categories do not exist in Empath 0.89:** grief, recreation, food, hunger, fatigue,
   tiredness, failure, success, confidence, cognitive_dissonance, cognitive_processes, thinking,
   motion, speed. Empath does not raise an error for an unknown category, so these score 0.0 without
   warning. Coverage is affected unevenly:
   - **C5** keeps only weakness and health.
   - **C8** keeps only movement.
   - **C7** keeps only confusion (present) and reading (absent).

   The mapping is kept as written so that the gap stays visible.
2. **Overlapping categories.** Some categories appear under more than one criterion: fatigue (C4,
   C5), health (C4, C5), weakness (C5, C6) and negative_emotion (C1, C6). Scores for those criteria
   are therefore not independent.
3. **No clinical review.** No psychologist or psychiatrist has evaluated the category choices or the
   directions.
4. **Role in the models.** In the legacy code, the mapping is used only for a descriptive
   correlation analysis (`legacy/X-Empath 2.ipynb`, cell 7). It does not enter any model:
   - The EmpathGCN graph takes its 150 category nodes from an ANOVA F-score ranking on the training
     split.
   - EmpathGCN's eight domain features come from a different hand-written list,
     `EMPATH_DOMAIN_GROUPS` (`legacy/Empath_GCN_Hybrid.ipynb`, cell 12).

   Any text that describes a model as "anchored in DSM-5" must match the code that produced the
   reported results.

## 6. How to verify

Check the category names against the installed Empath lexicon:

```python
import json
from empath import Empath

mapping = json.load(open("data/mapping/phq8_dsm5_empath.json", encoding="utf-8"))
lexicon = set(Empath().cats)
used = {c for crit in mapping["criteria"] for c in crit["present"] + crit["absent"]}
print(sorted(used - lexicon))  # should equal mapping["missing_from_empath"]
```

## References

- American Psychiatric Association. *Diagnostic and Statistical Manual of Mental Disorders*, 5th ed.
  (DSM-5). Arlington, VA: American Psychiatric Publishing, 2013.
- E. Fast, B. Chen, M. S. Bernstein. "Empath: Understanding Topic Signals in Large-Scale Text."
  *Proceedings of the CHI Conference on Human Factors in Computing Systems*, 2016.
- J. W. Pennebaker, R. L. Boyd, K. Jordan, K. Blackburn. *The Development and Psychometric
  Properties of LIWC2015*. Austin, TX: University of Texas at Austin, 2015.
