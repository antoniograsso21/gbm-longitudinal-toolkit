# FUTURE.md — Ideas for Future Versions

This file collects valid ideas that are out of scope for V1.
Do not move anything to the active backlog until V1 is complete.

---

## 1. Direct Use of Raw MRI Volumes

**Current approach**: radiomic features pre-extracted via PyRadiomics (CSV).

**Future approach**: use raw MRI volumes (NIfTI format, available on Figshare)
directly as input to deep learning models.

**Why it matters**:
- PyRadiomics features are hand-crafted — a CNN/ViT could learn richer representations
- Raw volumes enable spatial modelling beyond the segmented regions
- Self-supervised pre-training on longitudinal FLAIR volumes could improve
  label-efficiency on the small LUMIERE cohort

**Potential directions**:
- 3D CNN on longitudinal MRI volumes for RANO prediction
- Vision Transformer (ViT) on axial slices with temporal attention
- Self-supervised pre-training on FLAIR, fine-tune on RANO labels

**Prerequisites**:
- Download raw MRI from Figshare (~several GB)
- Visual QC of HD-GLIO-AUTO and DeepBraTumIA segmentations on raw volumes
- GPU infrastructure for 3D convolution training

---

## 2. Additional Graph Nodes

**Current V1 graph** (3 nodes — DeepBraTumIA):
```
Necrosis ←→ Contrast-enhancing
    ↖              ↗
         Edema
```

**Potential extensions**:
- **Surgical cavity**: relevant in post-operative scans, already visible in T1
- **White matter / deep grey matter**: infiltration quantification from FLAIR
- **Contralateral reference region**: normalisation target for intensity features

**Prerequisite**: V1 must first confirm that the 3-node graph adds value over
the 2-node baseline (ablation A6 in Phase 3). Additional nodes should be
motivated by empirical signal, not biological intuition alone.

---

## 3. Advanced Temporal Architectures for Irregular Time

Architectures that natively handle irregular continuous time:

- **Neural ODE**: models continuous-time dynamics, naturally handles irregular intervals
- **Temporal Point Process**: models event sequences with variable timing
- **Continuous-time RNN (CT-RNN)**: extends standard RNN to continuous time

**Prerequisite**: demonstrate empirically that sinusoidal delta_t encoding
is insufficient before adding complexity.

---

## 4. Multi-institutional Validation

**Potential datasets**:
- TCIA GBM collections (The Cancer Imaging Archive)
- BraTS longitudinal extension (when available)
- Prospective clinical data from collaborating institutions

**Prerequisite**: seek clinical collaboration AFTER V1, as stated in CONTEXT.md.

---

## 5. Prospective Clinical Integration

**Components needed**:
- FastAPI serving (already in tech stack as optional)
- DICOM integration for real-time feature extraction
- Prospective validation study with ethics approval

**Prerequisite**: V1 published and externally validated first.

---

## 6. Local SQLite Database for Data Exploration

**Why deferred**: on n=212 examples, Pandas handles all operations in memory
in milliseconds. The added complexity of a database outweighs any benefit.

**When it would make sense**: multi-institutional dataset (n > 10,000),
REST API serving, or concurrent multi-researcher access.

---

## 7. Intermediate Scans Without RANO as Additional Temporal Input

**Observation from audit**: several patients have MRI scans with extracted
radiomic features at timepoints that have no associated RANO label.
Example — Patient-024 has imaging at `week-018` and `week-019` but RANO
only at `week-015` and `week-031`. These intermediate scans are currently
discarded because the label shift requires a RANO anchor at both t and t+1.

**Potential value**: these unlabelled scans contain temporal signal about
tumour evolution between two labelled timepoints. Including them could
improve sequence-level representations, particularly for patients with
sparse RANO labels but dense imaging.

**Potential directions**:
- Semi-supervised sequence modelling: use unlabelled intermediate scans
  as additional GNN inputs within the temporal attention module, predicting
  RANO only at labelled anchor points
- Self-supervised pre-training on the full imaging sequence, fine-tuning
  on labelled pairs only
- Interpolation baseline: treat intermediate features as auxiliary inputs
  to the delta feature computation (finer-grained rate-of-change estimates)

**Prerequisite**: V1 must first establish baseline performance using
labelled-only pairs. The unlabelled scan count and patient coverage
should be quantified in a future audit extension.

**Note**: this does not affect V1 design. The current approach (use only
timepoints with RANO as sequence anchors) is the correct and defensible
choice for the first version.