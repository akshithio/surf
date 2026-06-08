# Goal

## Primary Target

The primary research target is an ICLR-strength main-conference submission.

A workshop submission can be an intermediate checkpoint for feedback, but it is not the ceiling. The project should be optimized around a real representation-learning contribution, not the minimum evidence needed for a workshop paper.

The intended paper is:

> A rigorous study of crop-temporal JEPA for irregular multimodal Earth-observation time series under domain shift. The paper should establish which temporal, missingness-aware, raw-cue-preserving, and patch-scale inductive biases improve agricultural transfer under realistic deployment conditions.

The project earns an ICLR submission by producing:

- a materially better crop-temporal JEPA architecture, not only a stricter evaluation protocol;
- matched comparisons against Presto, OlmoEarth, raw features, and relevant EO baselines;
- strict target-domain exclusion, sparse-label curves, sensor-dropout tests, and temporal-sparsity tests;
- a scaling study from CropHarvest point sequences to SSL4EO patch time series and then larger crop-season inputs;
- evidence on at least one crop benchmark beyond CropHarvest;
- clear separation of gains from objective, architecture, data scale, and input dimensionality.

## Scientific Question

> What inductive biases are needed for self-supervised representation learning on irregular multimodal agricultural time series under domain shift?

The key questions are:

1. Does temporal latent prediction learn transferable crop dynamics better than reconstruction-heavy objectives?
2. Does phenology-aware timing improve cross-region transfer?
3. Does missing-modality training improve robustness to sensor dropout and sparse observations?
4. Can raw-cue preservation retain red-edge, NIR, SWIR, NDVI, and radar signals that generic embeddings discard?
5. How much do unlabeled data scale, patch context, and longer crop-season sequences improve the representation?

## Model

### Current Prototype

The repaired CropHarvest path is point-level:

```text
N x T x C x 1 x 1
```

| Group | Channels |
|---|---|
| Sentinel-2 optical | `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`, `NDVI` |
| Sentinel-1 radar | `VV`, `VH` |
| Context | `temperature`, `precipitation`, `elevation` |
| Timing | DOY / month features |

The model emits an embedding:

```text
z = encoder(sequence)
```

Downstream probes use `z` for crop mapping and crop-type transfer. Stress, yield, and dense segmentation become legitimate downstream tasks only when the corresponding labels and patch inputs exist.

The current objective is a modest temporal JEPA prototype:

- modality encoders produce per-timestep features;
- a causal context encoder summarizes observations;
- an EMA target branch provides latent targets;
- a predictor learns target embeddings instead of reconstructing pixels;
- training augmentations drop sensors or timesteps.

This is useful diagnostic infrastructure. It is not the final architecture.

### Target Model

The target model should operate on patch time series:

```text
N x T x H x W x C
```

It should eventually include:

- spatial patches;
- actual acquisition dates;
- cloud and quality masks;
- S1/S2 availability masks;
- terrain and climate context;
- raw-cue-preserving targets;
- longer irregular crop-season sequences.

The immediate bridge is `[7]`:

```text
SSL4EO-S12 v1.1 pretraining: N x 4  x 16 x 16 x C
CropHarvest v2 evaluation:  N x 12 x 1  x 1  x C
```

The stronger later input should use raw Sentinel-derived crop-season chips:

```text
N x 12-36 x 16/32/64 x 16/32/64 x C
```

## Competitive Wedge

The project is not the first multimodal EO model, the first agriculture model, or the first JEPA for EO.

| Existing work | What it already owns |
|---|---|
| OlmoEarth, Galileo, TerraMind, Copernicus-FM | Broad multimodal EO representation learning at larger scale. |
| AnySat | JEPA-style EO learning across resolutions, scales, and modalities. |
| Presto | Lightweight pixel time-series modeling with missing-input handling and CropHarvest relevance. |
| CROMA, TerraFM, SSL4EO-S12 | Strong Sentinel-1 / Sentinel-2 fusion and scalable Sentinel SSL pipelines. |
| AgriFM | Agriculture-specific temporal pretraining at larger data scale. |
| AlphaEarth Foundations | Global annual embedding fields for sparse-label mapping. |

The useful open space is narrower:

1. crop-temporal JEPA designed around agricultural dynamics;
2. strict transfer with target-domain SSL exclusion;
3. sensor dropout and temporal sparsity as first-class evaluation axes;
4. phenology-aware timing and climate context;
5. raw-cue preservation for agronomic spectral and radar signals;
6. scaling from point sequences to crop-season patch sequences;
7. calibration-aware reporting across F1, AUROC, balanced accuracy, and thresholded behavior.

The strongest defensible claim would be:

> Broad EO foundation models are powerful, but crop monitoring requires temporal, missingness-aware, and agronomically faithful representation learning. SURF tests which crop-specific inductive biases and patch-time-series scaling choices improve transfer under realistic agricultural deployment conditions.

## Current Evidence

The current evidence is diagnostic, not a plateau.

| Result | Value | Interpretation |
|---|---:|---|
| Embedding-only strict priority F1 | `0.4915` | The point-level JEPA transfers, but it is not strong enough. |
| `[6]` hybrid JEPA + raw-stats priority F1 | `0.5307` | Raw-cue access materially improves transfer. |
| `[6]` hybrid AUROC | `0.6939` | Hybrid improves embedding-only AUROC by `+0.0284`. |
| OlmoEarth priority F1 / AUROC | `0.5553 / 0.6818` | OlmoEarth remains the strongest F1 pressure point; hybrid exceeds its AUROC. |
| Presto priority F1 / AUROC | `0.3845 / 0.7273` | Presto remains the ranking-quality pressure point. |

`[6]` is probe-only. It proves that the learned embedding is discarding useful raw information; it does not solve that failure inside the encoder.

LEM Brazil remains the key failure case. Hybrid improves it only mildly, while flattened raw inputs remain stronger. The missing information is not only compact statistics; some useful temporal or distributional raw structure is still being lost.

Early validation saturation in the CropHarvest point setup means the objective/data pair is too easy or too narrow. It does not show that JEPA, patch context, or larger data have plateaued.

## Immediate Roadmap

### `[7]`: Temporal Block-JEPA Screen

`[7]` is the first real architecture experiment.

Pretraining:

```text
data/processed/ssl4eo_s12_v11_48k.zarr
```

- `49,152` packaged SSL4EO-S12 v1.1 samples;
- four seasonal timesteps;
- `16 x 16` patches;
- exactly `768` source shards;
- stream-and-evict source cache capped at `60 GiB`;
- packaged SSL4EO screen, not a true agriculture-masked corpus.

Evaluation:

```text
data/cropharvest/processed/v2.zarr
```

The `[7]` arms isolate:

| Arm | Change |
|---|---|
| `A_control` | current causal next-embedding control |
| `B_full_target` | full temporal EMA target encoder |
| `C_transformer_predictor` | mask-conditioned Transformer predictor |
| `D_multiblock` | temporal multi-block masking |
| `E_cross_modal` | cross-modal latent prediction and dropout |
| `F_rawcue` | raw-cue auxiliary target |
| `G_full` | combined candidate |

`[7]` succeeds if embedding-only performance approaches or beats the `[6]` hybrid while shrinking the raw-feature gap, especially on LEM Brazil.

### After `[7]`

1. Expand the winning arm from the bounded SSL4EO screen to `100k`, then full or near-full SSL4EO.
2. Add an external crop benchmark such as EuroCropsML or ZueriCrop.
3. Build raw Sentinel crop-season chips with actual dates, masks, climate, and longer irregular sequences.
4. Run model-size and data-scale sweeps only after the objective and data path are stable.
5. Use large A100 allocation-scale compute only for validated designs.

## Scaling Plan

Potential compute may include roughly `22.5k` A100 GPU-hours. Treat that as a scaling budget for the best design, not exploration fuel.

| Stage | Data | Purpose |
|---|---|---|
| Diagnostic | CropHarvest v2 point tensors | Validate splits, probes, baselines, and failure cases. |
| First patch screen | `49,152` packaged SSL4EO samples | Select useful Block-JEPA ingredients. |
| Medium patch run | `100k` SSL4EO samples | Test data scaling. |
| Packaged-data scale | full or near-full SSL4EO | Train the strongest packaged Sentinel candidate. |
| Raw-chip scale | custom crop-season Sentinel chips | Test longer irregular temporal reasoning with real masks and climate. |
| Final scaled model | best validated recipe | Produce final comparisons and ablations. |

## ICLR Evidence Bar

The main submission needs:

1. Matched baselines:
   - raw features;
   - Presto;
   - OlmoEarth clean and verified stress baselines;
   - relevant EO baselines where input compatibility permits.
2. Robustness evaluation:
   - clean;
   - S2 off;
   - S1 off where supported;
   - climate off where supported;
   - temporal drop 50/70;
   - S2 off plus temporal drop;
   - label budgets `1%`, `5%`, `10%`, `25%`, and `100%`.
3. Method ablations:
   - next-step versus temporal block prediction;
   - MLP versus Transformer predictor;
   - phenology timing;
   - robustness augmentation;
   - cross-modal prediction;
   - raw-cue preservation.
4. Scale ablations:
   - CropHarvest point pretraining;
   - bounded SSL4EO patch pretraining;
   - larger SSL4EO subsets;
   - longer raw crop-season chips if ready.
5. Failure analysis:
   - LEM Brazil;
   - cue retention;
   - metric disagreement;
   - heldout and seed variance.
6. External validity:
   - at least one crop benchmark beyond CropHarvest;
   - explicit copied-versus-rerun labels for every baseline.

## Boundaries

Do not claim:

| Claim | Why not yet |
|---|---|
| Best EO foundation model | Current evidence is crop-focused and does not cover broad EO tasks. |
| Better than OlmoEarth overall | OlmoEarth remains broader and stronger on current default F1. |
| Full spatial foundation model | The current prototype is still `1 x 1`; `[7]` is only the first patch bridge. |
| Stress detection or yield prediction solved | Those require dedicated labels and evaluation. |
| Climate causality | Climate can be predictive without establishing causality. |
| No need for raw features | `[6]` and LEM Brazil show the opposite. |

## Experiment Selection Rule

Run an experiment if it answers at least one:

- Does it improve strict heldout transfer?
- Does it close the raw-cue, Presto AUROC, or OlmoEarth F1 gap?
- Does it improve LEM Brazil without damaging other heldouts?
- Does it isolate a JEPA architectural choice?
- Does it test robustness against a matched baseline?
- Does it expand data scale or input dimensionality with a clear hypothesis?
- Does it validate the result beyond CropHarvest?

Do not prioritize experiments that only improve random-split clean F1 or add architecture complexity without addressing a known failure mode.
