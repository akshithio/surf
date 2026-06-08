# SURF: JEPA Foundation Models for Agricultural Earth Observation

Primary Objective: Train a large-scale JEPA-style foundation model for agricultural EO (earth observation) and beat the current general-purpose and agriculture-specific remote sensing foundation models on established benchmarks. The target isn't a prototype but rather a SOTA representation learning for time-series, multi-sensor agricultural monitoring.

> #### A Foundation Model Approach to Crop Monitoring and Agro-ecosystem Analytics
>
> - Develop and evaluate a multimodal self-supervised representation learning pipeline for agricultural Earth observation.
> - Integrate spectral (Sentinel-2), radar (Sentinel-1), and climate variables at the pixel level.
> - Explore temporal prediction using next-embedding objectives and phenology-aware timing.
> - Assess model performance under real-world agricultural conditions: sparse labels, irregular sampling, and sensor dropout.
> - Demonstrate use for downstream agricultural tasks such as stress detection or yield-related proxies.
> - Document technical work, results, and research outcomes in a final written report.
>
> The selected student will work hands-on with multi-sensor satellite datasets to implement self-supervised learning methods and evaluate representation quality for agricultural applications. The student will experiment with spectral and radar data, integrate environmental variables, and assess generalization across temporal, spatial, and crop conditions. Additional activities include embedding visualization and interpretation, controlled experiments on sensor dropout and sparse temporal sampling, and communicating results through a poster / oral presentation and a technical report.

## Competitive Set

Primary competitors include:

- Prithvi-EO-2.0
- SSL4EO-S12 pretraining baselines
- AgriFM
- Galileo
- Presto / Lightweight, Pre-trained Transformers for Remote Sensing Timeseries
- SatMAE, Satlas, DOFA, U-TAE, PASTIS baselines, and other models used in those papers' benchmark tables

## Research Direction

Our contribution is centered on JEPA for multi-source agricultural time series:

- Learn predictive latent representations rather than reconstructing pixels alone.
- Handle Sentinel-1, Sentinel-2, HLS, MODIS, Landsat, climate, terrain, and other aligned covariates where benchmarks permit.
- Preserve temporal structure and phenology instead of flattening the problem into single-image classification.
- Evaluate against published SOTA on existing public benchmarks before claiming novelty.
- Scale training beyond the early CropHarvest experiments already in this repo.
- Use CropHarvest primarily as a strict transfer/probe benchmark once patch-scale SSL pretraining data is available.

## Repository Layout

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── checkpoints/
│   └── figures/
├── runners/
│   ├── [1].py
│   ├── [2].py
│   ├── [3].py
│   ├── [4].py
│   ├── [6].py
│   └── [7].py
├── notebooks/
│   ├── cropharvest.ipynb
│   ├── [1].ipynb
│   ├── [2].ipynb
│   ├── [3].ipynb
│   ├── [4].ipynb
│   ├── [5].ipynb
│   └── [6].ipynb
└── src/
```
