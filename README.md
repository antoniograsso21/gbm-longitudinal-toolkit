# gbm-longitudinal-toolkit

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**gbm-longitudinal-toolkit** is an open-source pipeline for longitudinal radiomics analysis in glioblastoma (GBM), combining temporal graph modelling with distribution-free uncertainty quantification to study treatment response and disease evolution over time.

## Dataset

The toolkit assumes access to:
- **Longitudinal MRI** scans (multi-timepoint, optionally multi-sequence)
- **Lesion/region annotations** or segmentation masks per timepoint
- **Clinical metadata** (e.g., treatment dates, survival, molecular markers)

Dataset-specific loading and preprocessing hooks are designed to be configurable; example configurations and data schemas will be added as the project matures.

## Architecture

At a high level, the toolkit provides:
- **Preprocessing & registration**: temporal alignment, intensity normalization, and ROI handling.
- **Radiomics feature extraction**: reproducible feature computation across timepoints.
- **Temporal graph modelling**: graph-based representations of longitudinal trajectories.
- **Uncertainty-aware inference**: distribution-free methods to quantify prediction reliability.
- **Experiment harness**: configuration-driven pipelines for training, evaluation, and ablations.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/gbm-longitudinal-toolkit.git
cd gbm-longitudinal-toolkit

# (Recommended) create and activate a Python 3.12+ virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .