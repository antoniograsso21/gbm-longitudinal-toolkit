# FUTURE.md — Ideas for Future Versions

This file collects valid ideas that are out of scope for V1.
Do not move anything to the active backlog until V1 is complete.

---

## 1. Third Graph Node: Peritumoral Edema

**Idea**: add a third node to the graph to explicitly model the peritumoral
compartment (edema/infiltration), extractable from the FLAIR sequence
already present in LUMIERE.

**Current graph** (2 nodes):
```
ET ←→ NC
```

**Extended graph** (3 nodes, triangular topology):
```
ET ←→ NC
 ↖      ↗
  Edema
```

**Biological rationale**: tumor infiltration into the peritumoral region is one
of the strongest predictors of progression in GBM. Many radiomics studies show
that peritumoral FLAIR features add independent predictive signal beyond ET and NC.

**Prerequisites**:
- Verify edema segmentation availability in LUMIERE
  (HD-GLIO-AUTO segments ET and NC — evaluate if edema is recoverable
  as the difference between FLAIR mask and lesion mask)
- Re-evaluate feature extraction pipeline (PyRadiomics on edema region)
- Re-evaluate scientific claim (3 nodes removes the main declared limitation)

**Reference**: to be included in the Future Work section of the V1 paper
as a natural framework extension.

---

## 2. Direct Use of Raw MRI Volumes

**Current approach**: radiomic features pre-extracted via PyRadiomics (CSV).

**Future approach**: use raw MRI volumes (NIfTI format, available on Figshare)
directly as input to deep learning models.

**Why it matters**:
- PyRadiomics features are hand-crafted — a CNN/ViT could learn richer representations
- Raw volumes enable spatial modeling beyond the two segmented regions
- Peritumoral edema segmentation could be automated directly on FLAIR volumes

**Potential directions**:
- 3D CNN on longitudinal MRI volumes for RANO prediction
- Vision Transformer (ViT) on axial slices with temporal attention
- Self-supervised pre-training on FLAIR volumes, fine-tune on RANO labels

**Prerequisites**:
- Download raw MRI from Figshare (~several GB)
- Verify NIfTI format consistency across patients and timepoints
- Visual QC of HD-GLIO-AUTO segmentations on raw volumes
- GPU infrastructure for 3D convolution training

**Note**: this direction requires significantly more compute and data infrastructure
than V1. Suitable for a follow-up paper or extended version of the framework.

---

## 3. Advanced Temporal Architectures for Irregular Time

LUMIERE is irregularly longitudinal (non-fixed intervals).
Architectures that natively handle irregular continuous time:

- **Neural ODE** (Chen et al. 2018): models continuous-time dynamics,
  naturally handles irregular intervals
- **Temporal Point Process**: models event sequences with variable timing
- **Continuous-time RNN (CT-RNN)**: extends standard RNN to continuous time

**Prerequisite**: first demonstrate empirically that LSTM/GNN are insufficient
for this specific aspect — do not add complexity without empirical evidence.

---

## 4. Multi-institutional Validation

**Idea**: validate the framework on an external GBM longitudinal dataset
beyond LUMIERE to demonstrate generalizability.

**Potential datasets**:
- TCIA GBM collections (The Cancer Imaging Archive)
- BraTS longitudinal extension (when available)
- Prospective clinical data from collaborating institutions

**Prerequisite**: seek clinical collaboration AFTER V1, as stated in CONTEXT.md.

---

## 5. Prospective Clinical Integration

**Idea**: deploy the framework as a clinical decision support tool,
integrated into a radiology workflow.

**Components needed**:
- FastAPI serving (already in tech stack as optional)
- DICOM integration for real-time feature extraction
- Prospective validation study with ethics approval

**Prerequisite**: V1 published and externally validated first.

---

---

## 6. Local SQLite Database for Data Exploration

**Idea**: load all LUMIERE CSVs into a local SQLite database to enable
declarative SQL queries during EDA and preprocessing development.

**Why it was deferred**: on n=318 examples, Pandas handles all joins,
filters, and aggregations in memory in milliseconds. The added complexity
of a database connection, schema definition, and migration management
outweighs any benefit at this scale.

**When it would make sense**:
- Dataset grows significantly (multi-institutional, n > 10,000)
- A REST API (FastAPI) needs to serve predictions and query results
- Multiple researchers need concurrent read access to processed data

**Reference**: FastAPI + SQLite is a standard lightweight pattern for
serving ML results without a full database infrastructure.
