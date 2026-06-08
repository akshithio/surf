# Steps

This file is the operational priority list. Update it after every experiment with:

- what the last experiment changed;
- what failed or became uncertain;
- what the next experiment must isolate;
- which longer-term goals are now closer or blocked.

## Current State

`[10]` is the immediate run.

It is not a final scale run. It is a controlled architecture and evaluation run on the current SSL4EO `48k + 48k` materialized stores, with EuroCropsML added as the second benchmark. The purpose is to decide which recipe is worth scaling or modifying, not to prove the ceiling of the project.

Current `[10]` questions:

1. Does S1-off training improve missing-sensor robustness without hurting S2-off robustness?
2. Does mixed generic/agriculture SSL4EO data help beyond update exposure or sample count?
3. Can spatial JEPA avoid the trivial/position-only failure seen in earlier spatial arms?
4. Does variance/covariance content regularization improve spatial representation quality?
5. Do learned embeddings transfer to EuroCropsML under Latvia/Portugal to Estonia crop-type transfer?

## Immediate Priority

Run `[10]`, then analyze it before changing architecture again.

Required first readout:

- final CropHarvest strict holdout table by arm, seed, condition, and model type;
- EuroCropsML macro-F1, balanced accuracy, effective label budget, and unseen-class drops;
- checkpoint diagnostics: normal loss, content-shuffle gap, missingness-shuffle gap, effective rank, same-slot cosine, clean-zero cosine;
- corruption diagnostics for S2-off, S1-off, temporal-drop, and paired sensor/time-drop conditions;
- whether best behavior appears at final checkpoint or earlier checkpoints.

The decision should not be based only on JEPA loss. Near-zero loss with weak content-shuffle separation is a failed representation.

## Likely Next Experiments

### If `[10]` Fails Because Spatial Tokens Still Collapse

Run a narrow anti-collapse experiment before scaling data.

Candidate `[11]`:

| Arm | Change | Purpose |
|---|---|---|
| baseline | best `[10]` arm unchanged | reference |
| stronger variance/covariance | tune content regularizer only | test whether collapse is regularization strength |
| EC-style positional conditioning | context encoder sees target positions; target encoder sees context positions | test whether shortcut is positional |
| SIGReg add-on | add SIGReg to pooled/global embeddings while keeping EMA | test explicit distribution regularization |

Do not remove EMA in the first SIGReg run. First test whether SIGReg helps as an additive regularizer. Removing EMA should be a second step only if additive SIGReg improves diagnostics and transfer.

### If `[10]` Transfers But Sensor-Drop Robustness Is Weak

Prioritize cross-modal JEPA.

Candidate `[11]`:

| Arm | Change | Purpose |
|---|---|---|
| baseline | best `[10]` pooled or spatial arm | reference |
| S2-from-S1/context | predict S2 latent target from S1 plus timing/context | direct cloud-gap training |
| S1-from-S2/context | predict S1 latent target from S2 plus timing/context | radar-gap symmetry |
| bidirectional cross-modal | both terms | test whether shared representation improves |

This is the highest-value JEPA-space idea if `[10]` shows the architecture is otherwise healthy.

### If `[10]` Is Strong On CropHarvest But Weak On EuroCropsML

Treat it as a transfer/generalization problem, not a leaderboard problem.

Next actions:

- audit EuroCropsML preprocessing and label taxonomy;
- check whether raw stats beat embeddings by crop group;
- inspect class imbalance, unseen classes, and country-specific label gaps;
- add UMAP by crop class, country, and seasonal coverage;
- test whether the representation is clustering by geography instead of phenology/crop type.

Only after this should we change the model.

### If `[10]` Is Healthy Across Both Benchmarks

Do not immediately jump to huge scale unless the winning recipe is clean.

Recommended next steps:

1. rerun the winning arm with one focused JEPA modification: cross-modal loss or SIGReg add-on;
2. add a raw-cue auxiliary target if LEM Brazil or red-edge-sensitive classes remain weak;
3. then build a larger SSL4EO corpus and run the best recipe at higher data scale.

## JEPA Architecture Backlog

Priority order after `[10]`:

1. X-JEPA-style cross-modal prediction: S1/context to S2 latent and optionally S2/context to S1 latent.
2. EC-IJEPA-style positional conditioning if spatial shortcuts persist.
3. DMT/raw-cue target: preserve red-edge, NIR, SWIR, NDVI, and radar statistics inside the representation.
4. SIGReg add-on: test as a collapse/conditioning regularizer before any no-EMA variant.
5. MTS-style trend/shock objective: split seasonal phenology from transient stress/cloud/noise events.
6. M3-style modality-routed predictor: use only if shared predictor struggles across missing-modality regimes.
7. SALT/static-teacher target: use only if EMA targets stay noisy, smooth, or unstable after simpler fixes.

Do not prioritize NEPA while frozen linear probing is the main evaluation protocol. It is better aligned with fine-tuning-heavy evaluation.

## Climate Reintegration

Climate is important, but it should return only when the join is real.

Planned progression:

1. Keep `[10]` climate-free for SSL4EO pretraining.
2. Use climate in evaluation only where already valid.
3. Build a real temporal climate join for SSL4EO or raw Sentinel crop chips.
4. Test climate as:
   - context-only input;
   - predicted target;
   - dropout modality;
   - phenology timing aid, possibly through growing-degree-day features.

Do not add repeated or synthetic climate channels to the SSL4EO store and call it climate-aware pretraining.

## Data Scale

The current `48k + 48k` SSL4EO corpus is acceptable for recipe selection. It is not the final scale.

Scale only after:

- preprocessing is stable;
- EuroCropsML is integrated;
- spatial diagnostics are healthy;
- at least one objective/architecture change clearly improves transfer or robustness.

Next scale targets after the recipe is stable:

| Stage | Materialized target | Purpose |
|---|---:|---|
| current | about `3-4 GB` | architecture screen |
| medium | `15-25 GB` | first data-scale curve |
| large | `40-60 GB` | serious pretraining corpus |
| allocation-scale | larger raw-chip or full packaged corpus | final scaled model |

If the model fails at current scale because of objective issues, scaling is premature. If it succeeds but remains data-limited, scaling becomes the next clean experiment.

## Interpretation Work

These are not blockers for launching `[10]`, but they are required for a serious paper path.

1. UMAP by crop class, country, and seasonal coverage on EuroCropsML.
2. UMAP by CropHarvest holdout group and crop/non-crop label.
3. Raw-cue retention probes for red-edge/NIR/SWIR/NDVI and VV/VH.
4. Phenological residual probe: does the embedding capture growth-rate deviation rather than only absolute greenness?
5. Calibration report: AUROC, default F1, calibrated F1, balanced accuracy, and threshold behavior.

## Current Non-Goals

- Do not build a bigger model only because current experiments feel small.
- Do not train on a 60 GB corpus until the recipe is worth scaling.
- Do not make climate-aware claims without a real climate join.
- Do not treat lower SSL loss as success without transfer and content diagnostics.
- Do not add many JEPA variants in one run; each experiment should isolate one mechanism.
