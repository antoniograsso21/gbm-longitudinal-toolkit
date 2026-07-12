# Step 8 — Paper ⏳

## Objective
Write and submit the bioRxiv preprint.

**Target**: bioRxiv preprint → Medical Image Analysis / NeuroImage / Scientific Reports

---

## Paper Structure

**Abstract** (250 words)
- Clinical problem: RANO assessment is subjective and delayed
- What we built: longitudinal radiomics pipeline with 3-node GNN + UQ
- Key result: macro F1, comparison to baselines, CP coverage
- Claim: reusable framework for any longitudinal radiomics dataset

**1. Introduction**
- GBM clinical context
- Limitations of current radiomics papers: cross-sectional, no UQ, not reusable
- What this paper contributes: longitudinal, 3-node graph, calibrated, open-source
- Literature review: verify claim before writing

**2. Methods**
- Dataset: LUMIERE, DeepBraTumIA segmentation, n_effective=231, 64 patients
- Data quality: 39 scans absent from CSV, 70 all-NaN scans, 63 any-NaN dropped
- Patient-025 exclusion, Patient-039 loss, Patient-042 duplicate resolution
- Preprocessing pipeline (Step 1)
- Feature engineering (Step 2)
- Feature selection: mRMR + Stability Selection inside CV (Step 3)
- Baseline hierarchy (Step 3)
- Graph construction: 3-node triangular topology (Step 4)
- Temporal GNN architecture (Step 4)
- Conformal Prediction setup (Step 6)
- Interpretability methods (Step 5)
- Cross-validation: StratifiedGroupKFold, normalization inside fold

**3. Results**
- Comparison table (all models, all metrics)
- Ablation study results (A1–A6 from Step 4 + A/B/C/D from Step 3)
- Conformal Prediction coverage and set sizes
- Feature importance: SHAP (LightGBM) vs IG (GNN) — do they agree?
- Attention weight analysis per disease stage

**4. Discussion**
- Does GNN beat LightGBM? Honest interpretation either way.
- Does the 3rd node (Edema) add value? Honest result either way.
- Clinical relevance of prediction sets
- Limitations: single institution, inter-rater variability unknown,
  automated segmentation not QC'd, n=64 (small Response class),
  mean sequence length ~3.6 timepoints

**5. Future Work** — reference FUTURE.md

---

## Definition of Done

- [ ] Literature review complete
- [ ] All 5 sections drafted
- [ ] All figures and tables final
- [ ] bioRxiv submission
- [ ] GitHub release tagged to match preprint version
