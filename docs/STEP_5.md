# Step 5 — Interpretability ⏳

## Objective
Produce clinically meaningful explanations at three levels.
Requires trained models from Step 3 (LightGBM) and Step 4 (GNN).

---

## Level 1 — Global Feature Importance
Feature stability scores from mRMR + Stability Selection (Step 3).
Ranked table grouped by region (CE/ED/NC) and feature family.
SHAP beeswarm from LightGBM (already computed in Step 3).

## Level 2 — Attention Weights
Direct outputs of GATv2Conv and temporal attention — no additional libraries.

- **Temporal attention**: which timepoint matters most per patient?
  Report mean weight profile: baseline scan vs most recent scan.
- **GATv2 edge attention**: which inter-compartment relationship dominates?
  Report mean per disease stage. Do Response patients attend more to CE↔ED?

Caution: attention ≠ causation. Report as "where the model focused", not "what caused the prediction".

## Level 3 — Integrated Gradients (per-patient)
```python
from captum.attr import IntegratedGradients
ig = IntegratedGradients(model)
attributions = ig.attribute(inputs=patient_graphs, baselines=zero_graphs, target=pred_class)
```
Top-5 features by attribution magnitude + most predictive timepoint per patient.

**File**: `src/interpretability/integrated_gradients.py`

## Clinical Summary Output (combined with Step 6)
```
Patient: Patient-XXX  |  Scan: week-NNN
Predicted: Progressive  |  CP set (95%): {Progressive, Stable}
Top features: NC_T1_shape_MeshVolume (+0.43), CE_FLAIR_firstorder_Mean (+0.31)
Most predictive timepoint: week-137 (attention weight=0.62)
```
**File**: `src/interpretability/clinical_summary.py`

---

## Definition of Done

- [ ] Feature stability table saved to `experiments/`
- [ ] Temporal + edge attention figures saved
- [ ] Integrated Gradients implemented and tested on ≥3 patients (one per class)
- [ ] Clinical summary output format implemented
- [ ] Check: do low-confidence CP sets correlate with diffuse IG attributions?
