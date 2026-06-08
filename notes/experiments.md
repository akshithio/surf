# Experiments

This log keeps durable experiment context only: what each experiment tested, why it was run, what result would matter, what happened, what was flawed, and what the next move should be. Ephemeral run details such as temporary process ids, transient shell state, and short-lived log paths belong in local terminal history, not here.

## [1] CropHarvest Temporal JEPA v0

### Overview

First full CropHarvest run of a leak-free temporal JEPA model. The model used a causal temporal context encoder, an EMA target encoder, Sentinel-1, Sentinel-2, climate variables, and day-of-year features. Evaluation used frozen embeddings with logistic-regression probes over random stratified splits and stress conditions. The earlier bidirectional next-embedding direction could leak future timesteps into the representation. This run tested whether a causal/EMA JEPA setup was worth scaling at all.

### What We Hoped To See
- JEPA frozen embeddings beat raw flattened features.
- Gains are strongest under sparse labels and sensor/time corruption.
- Training loss does not collapse immediately.

### Results
- JEPA beat raw flattened features across the main conditions.
- Gains were strongest under missing modalities and temporal drops.
- JEPA priority F1 / AUROC: `0.7734` / `0.7880`.
- Hybrid JEPA+raw-stats F1 / AUROC: `0.7831` / `0.7952`.
- Calibrated JEPA F1: `0.8338`.
- The self-supervised validation loss stopped improving early, around epoch `5-7`, so `100` epochs was wasteful.

The signal was real enough to continue. The method improved label efficiency and robustness, but the training schedule was too long for the size of the data.

### Issues
- Random split was too forgiving.
- The dataset was still only CropHarvest point-level time series.
- No external baseline was reproduced on the same protocol.

### Next

Run a breadth-oriented screen over seeds, robustness settings, DOY, learning rate, and model size instead of simply increasing epochs.

## [2] CropHarvest JEPA v1 Screen

### Description

Experiment `[1]` showed the method was promising but overtrained quickly. The next question was whether the gain was stable and which ingredients mattered. Screened multiple JEPA variants over CropHarvest with shorter training, multiple seeds, robustness ablations, lower learning rate, and a larger medium model. The priority score averaged F1 across clean, S2-off, temporal-drop, and combined-stress conditions.

### What We Hoped To See
- Stable JEPA gains over seeds.
- Robustness augmentations improve stress performance.
- DOY helps or at least does not hurt.
- A slightly larger model improves clean and stress performance.

### Results
- `medium_low_lr` became the mainline.
- It had the best three-seed priority F1 / AUROC: `0.7755` / `0.7899`.
- Calibrated F1 reached `0.8336`.
- `no_robust` looked decent clean but degraded more under combined stress.
- `no_doy` was the weakest overall configuration.

### Interpretation

The JEPA gain was stable enough to move forward. Robustness augmentations should stay. DOY should stay. Longer training was not useful; best checkpoints appeared early.

### Issues
- Still random-split CropHarvest.
- Still no external baseline under the same protocol.
- The result could reflect local interpolation rather than genuine dataset transfer.

### Next

Evaluate grouped dataset holdouts using the `dataset` property from CropHarvest labels.

## [3] CropHarvest JEPA v2 Confirm Generalization

### Description

Ran larger JEPA configurations and grouped-holdout probes. The grouped probe trained the frozen probe on all datasets except one and tested on that heldout dataset, while the SSL encoder could still see heldout-domain unlabeled samples.

### Motivation

Random splits were not enough evidence. The goal was to see whether representations transferred across CropHarvest source datasets.

### What We Hoped To See

- Larger models improve clean random-split performance.
- Representations still transfer under dataset-heldout probes.
- Robust pretraining improves sensor/time failure modes.

### Results

- `large_dual_s2` became the strongest mainline by F1.
- Random-split priority F1 reached `0.7869`.
- Clean full-label F1/AUROC reached `0.8289 / 0.8764`.
- Best grouped-holdout (medium-screen config `no_doy`): F1 `0.4959`, AUROC `0.6525`.
- Best grouped calibrated F1 (`medium_low_lr`): `0.5814`.
- Grouped holdout was much harder than random split.

### Interpretation

Scaling helped, but the real difficulty was transfer across datasets. The result justified a stricter test where the heldout dataset is excluded from SSL pretraining as well.

### Issues

- The encoder may have seen unlabeled heldout-domain samples during SSL.
- Raw grouped baselines were not yet computed as carefully as needed.
- F1 was unstable under dataset shift, so AUROC and balanced accuracy needed more emphasis.

### Next

Run strict dataset-heldout SSL: exclude the heldout dataset from SSL pretraining, probe training, and test only on the heldout dataset.

## [4] Strict Dataset-Heldout JEPA

### Description

Strict heldout evaluation for `large_dual_s2` and `large_default`. For each heldout dataset, the heldout group was excluded from SSL pretraining and from probe training, then tested only on that dataset. Heldouts were `rwanda-ceo`, `togo`, `togo-eval`, `ethiopia`, and `lem-brazil`.

### Motivation

Grouped holdout was useful but did not answer whether unlabeled target-domain exposure was carrying the result. This experiment tested true cross-dataset representation transfer.

### What We Hoped To See

- JEPA remains above raw features under strict transfer.
- `large_dual_s2` improves robustness relative to `large_default`.
- The strict protocol exposes which datasets are genuine bottlenecks.

### Results

Strict priority aggregate:

| Model | Priority F1 | Priority AUROC | Calibrated F1 |
|---|---:|---:|---:|
| `large_dual_s2_jepa` | 0.4917 | 0.6655 | 0.5839 |
| `large_default_jepa` | 0.4877 | 0.6674 | 0.5854 |
| `raw_flattened` | 0.3112 | 0.6111 | — |

Heldout priority F1:

| Heldout | `large_dual_s2` | `large_default` | Raw |
|---|---:|---:|---:|
| `ethiopia` | 0.6847 | 0.6893 | 0.3336 |
| `lem-brazil` | 0.3307 | 0.3374 | 0.3605 |
| `rwanda-ceo` | 0.4317 | 0.4201 | 0.2662 |
| `togo` | 0.5320 | 0.5186 | 0.3161 |
| `togo-eval` | 0.4911 | 0.4834 | 0.2794 |

### Interpretation

Strict heldout drops hard relative to random split, but not meaningfully relative to the earlier clean grouped-holdout probes. That means unlabeled target-domain exposure probably was not carrying the grouped result. The real bottleneck is cross-dataset labeled transfer, especially under stress.

JEPA still beat raw overall by about `+0.18` F1. The sharp failure was `lem-brazil`, where raw features beat both JEPA variants by F1 and AUROC. Calibrated F1 improved substantially over default F1 for both JEPA variants (+0.09), while raw calibration had no effect because raw probes already used source-validation thresholds.

### Issues

- LEM Brazil showed that the learned representation can lose simple transferable class information.
- `large_dual_s2` and `large_default` were close enough that over-weighting either one would be premature.
- No external baseline was yet on this exact strict protocol.

### Next

Run Presto and OlmoEarth on the same strict protocol, then diagnose LEM Brazil.

## [5] External Strict-Heldout Baselines And LEM Brazil Diagnostic

### Description

Reproduced `Presto` and `OlmoEarth` frozen-embedding baselines on the same strict heldout protocol as `[4]`. Also ran a LEM Brazil diagnostic over class balance, raw feature separability, and raw predictions.

### Motivation

The key question after `[4]` was whether LEM Brazil is a general learned-representation failure or a failure specific to our JEPA.

### What We Hoped To See

- External baselines clarify how far our representation is from the field.
- If LEM breaks all learned embeddings, the next move is representation design.
- If LEM only breaks our model, the next move is model-specific debugging.

### Results

Clean strict-transfer aggregate:

| Baseline | Clean F1 | Clean AUROC | Balanced Acc | Calibrated F1 |
|---|---:|---:|---:|---:|
| `olmoearth` | 0.5996 | 0.7296 | 0.6732 | 0.6124 |
| `presto` | 0.4588 | 0.7621 | 0.6360 | 0.6172 |

Priority aggregate across `clean`, `sensor_off_s2`, `temporal_drop_50`, `temporal_drop_70`, and `s2_off_tdrop50`:

| Baseline | Priority F1 | Priority AUROC | Balanced Acc | Calibrated F1 |
|---|---:|---:|---:|---:|
| `olmoearth` | 0.5553 | 0.6818 | 0.6256 | 0.5782 |
| `large_dual_s2_jepa` | 0.4917 | 0.6655 | 0.6068 | 0.5839 |
| `large_default_jepa` | 0.4877 | 0.6674 | 0.6063 | 0.5854 |
| `presto` | 0.3845 | 0.7273 | 0.6027 | 0.5756 |
| `raw_flattened` | 0.3181 | 0.6100 | 0.5444 | 0.5464 |

LEM Brazil diagnostic:

| Baseline | Protocol | F1 | AUROC | Balanced Acc |
|---|---|---:|---:|---:|
| `raw_flattened` | priority stress aggregate | 0.3598 | 0.6500 | 0.5872 |
| `presto` | priority stress aggregate | 0.3415 | 0.5611 | 0.5487 |
| `large_default_jepa` | priority stress aggregate | 0.3327 | 0.5699 | 0.5624 |
| `olmoearth` | priority stress aggregate | 0.3273 | 0.5451 | 0.5464 |
| `large_dual_s2_jepa` | priority stress aggregate | 0.3261 | 0.5624 | 0.5571 |

Top LEM raw separability cues:

| Feature | Statistic | LEM abs AUROC | Source abs AUROC |
|---|---|---:|---:|
| `B8A` | mean | 0.8221 | 0.5103 |
| `B8` | mean | 0.8170 | 0.5032 |
| `B7` | mean | 0.8150 | 0.5064 |
| `B9` | mean | 0.7936 | 0.5710 |
| `NDVI` | mean | 0.7515 | 0.5084 |

### Interpretation

`OlmoEarth` is the strongest external baseline by clean F1 and priority F1 after fixing the stress path. Its metrics now change across stress conditions, so the previous identical-embedding failure is resolved. `Presto` remains the strongest external baseline by AUROC, so calibration and thresholding matter under strict transfer. Our JEPA beats Presto by default F1 and narrowly beats both external baselines by calibrated F1, but loses to `OlmoEarth` by default priority F1 and loses to `Presto` by AUROC.

LEM Brazil is not only a SURF-specific failure. Presto, OlmoEarth, and our JEPA all fail to beat raw there under the priority stress aggregate. The likely reason is simple optical/red-edge/NIR magnitude structure: raw `B8A`, `B8`, `B7`, and `NDVI` statistics separate LEM crops well, but those cues are not globally stable in the source pool and are not preserved cleanly by learned embeddings.

### Issues

- External baselines were run as frozen embeddings, not exact full fine-tuning reproductions.
- The LEM diagnostic points to a representation issue but does not yet fix it.
- The remote postprocess step for the fixed OlmoEarth stress run failed because `pandas` was missing in that shell, but the OlmoEarth runner itself exited successfully and the combined tables were rebuilt locally.

### Next

Use the corrected strict-transfer and external-baseline evidence from `[4]` and `[5]` to decide the next active experiment. Do not treat the raw-cue long run as part of the canonical experiment sequence.

## [6] Strict Hybrid Raw-Cue Probe

### Description

Probe-only strict heldout experiment over existing repaired `[4]` checkpoints. This experiment does not train a new encoder. It loads `large_default` and `large_dual_s2` strict checkpoints for seeds `7`, `11`, and `42`, evaluates only the checkpoint's matching strict heldout dataset, and compares four probe inputs:

| Arm | Meaning |
|---|---|
| `surf_jepa_v0` | Frozen JEPA embedding only. |
| `raw_stats` | Compact raw spectral/SAR summary statistics only. |
| `surf_jepa_v0_plus_raw_stats` | JEPA embedding concatenated with raw statistics. |
| `raw_flattened` | Flattened raw time series baseline for continuity. |

### Motivation

LEM Brazil showed that raw `B8A`, `B8`, `B7`, and `NDVI` statistics separate the heldout classes better than every learned embedding. Before training a raw-cue loss, distilling from a teacher, or scaling pretraining, this experiment asks whether simply exposing those simple spectral magnitude cues at probe time closes the strict-transfer gap.

Core question:

> Is the gap to OlmoEarth/Presto because the JEPA encoder is bad, or because the learned embedding discards simple spectral magnitude cues that still matter under strict transfer?

### What We Hope To See

- Hybrid JEPA plus raw stats closes much of the priority F1 gap to OlmoEarth.
- Hybrid AUROC moves materially toward Presto.
- LEM Brazil improves without reducing non-LEM heldout performance.
- Raw stats help under stress, not only on clean inputs.

### Success Criteria

The hybrid arm is meaningful if it satisfies at least one:

- `surf_jepa_v0_plus_raw_stats` priority F1 is at least `0.53`.
- Hybrid AUROC improves by at least `0.025` over embedding-only JEPA.
- LEM Brazil improves without reducing non-LEM priority performance.
- Stress-condition gains appear beyond clean-only gains.

### Results

The table-grade rerun used stable `liblinear` probe settings and produced zero convergence warnings.

The run evaluated all `30` strict checkpoints and produced `3000` probe rows.

Priority aggregate:

| Config | Arm | Priority F1 | Priority AUROC | Balanced Acc | Calibrated F1 | Calibrated Balanced Acc |
|---|---|---:|---:|---:|---:|---:|
| `large_default` | `raw_flattened` | 0.3178 | 0.6100 | 0.5444 | 0.5466 | 0.5334 |
| `large_default` | `raw_stats` | 0.5245 | 0.6581 | 0.6118 | 0.5643 | 0.5300 |
| `large_default` | `surf_jepa_v0` | 0.4874 | 0.6674 | 0.6063 | 0.5855 | 0.5847 |
| `large_default` | `surf_jepa_v0_plus_raw_stats` | 0.5287 | 0.6957 | 0.6328 | 0.5920 | 0.5970 |
| `large_dual_s2` | `raw_flattened` | 0.3178 | 0.6100 | 0.5444 | 0.5466 | 0.5334 |
| `large_dual_s2` | `raw_stats` | 0.5245 | 0.6581 | 0.6118 | 0.5643 | 0.5300 |
| `large_dual_s2` | `surf_jepa_v0` | 0.4915 | 0.6655 | 0.6067 | 0.5839 | 0.5819 |
| `large_dual_s2` | `surf_jepa_v0_plus_raw_stats` | 0.5307 | 0.6939 | 0.6318 | 0.5911 | 0.5961 |

LEM Brazil priority aggregate:

| Config | Arm | Priority F1 | Priority AUROC | Balanced Acc | Calibrated F1 | Calibrated Balanced Acc |
|---|---|---:|---:|---:|---:|---:|
| `large_default` | `raw_flattened` | 0.3599 | 0.6500 | 0.5873 | 0.3204 | 0.5028 |
| `large_default` | `raw_stats` | 0.3293 | 0.5894 | 0.5635 | 0.3345 | 0.5313 |
| `large_default` | `surf_jepa_v0` | 0.3326 | 0.5699 | 0.5623 | 0.3475 | 0.5563 |
| `large_default` | `surf_jepa_v0_plus_raw_stats` | 0.3456 | 0.5777 | 0.5683 | 0.3462 | 0.5545 |
| `large_dual_s2` | `raw_flattened` | 0.3599 | 0.6500 | 0.5873 | 0.3204 | 0.5028 |
| `large_dual_s2` | `raw_stats` | 0.3293 | 0.5894 | 0.5635 | 0.3345 | 0.5313 |
| `large_dual_s2` | `surf_jepa_v0` | 0.3261 | 0.5624 | 0.5572 | 0.3460 | 0.5539 |
| `large_dual_s2` | `surf_jepa_v0_plus_raw_stats` | 0.3402 | 0.5715 | 0.5623 | 0.3461 | 0.5541 |

Comparison to external baselines from `[5]`:

| Model | Priority F1 | Priority AUROC | Calibrated F1 |
|---|---:|---:|---:|
| `olmoearth` | 0.5553 | 0.6818 | 0.5782 |
| `presto` | 0.3845 | 0.7273 | 0.5756 |
| best `[6]` hybrid | 0.5307 | 0.6957 | 0.5920 |

For `large_dual_s2`, hybrid improves every priority condition relative to embedding-only:

| Condition | JEPA F1 | Hybrid F1 | JEPA AUROC | Hybrid AUROC |
|---|---:|---:|---:|---:|
| `clean` | 0.5067 | 0.5395 | 0.6841 | 0.7080 |
| `sensor_off_s2` | 0.5003 | 0.5502 | 0.6958 | 0.7377 |
| `temporal_drop_50` | 0.4987 | 0.5324 | 0.6639 | 0.6855 |
| `temporal_drop_70` | 0.4920 | 0.5245 | 0.6469 | 0.6684 |
| `s2_off_tdrop50` | 0.4600 | 0.5070 | 0.6368 | 0.6699 |

### Interpretation

Hybrid helps enough to treat raw-cue preservation as real. `large_dual_s2` hybrid reaches priority F1 `0.5307`, clearing the planned `0.53` threshold, and improves AUROC by `+0.0284` over embedding-only JEPA. The same pattern appears under stress, including `sensor_off_s2` and `s2_off_tdrop50`, so this is not just a clean-condition artifact.

The result does not mean raw stats alone solve the problem. Raw stats beat embedding-only by default priority F1 (`0.5246` vs `0.4917`) but lag on AUROC (`0.6581` vs `0.6655`) and calibrated F1 (`0.5643` vs `0.5839`). The hybrid arm is the best overall SURF-side representation in this table because it keeps the JEPA geometry while restoring some simple spectral/SAR summary cues.

LEM Brazil improves only mildly. For `large_dual_s2`, hybrid raises LEM F1 from `0.3261` to `0.3402` and AUROC from `0.5624` to `0.5715`, but raw flattened still wins LEM with F1 `0.3599` and AUROC `0.6500`. That means probe-time raw stats are useful but do not fully recover the raw temporal/magnitude signal that LEM depends on.

### Issues

- This is probe-only. It shows that raw-cue access helps downstream probes, not that the encoder itself preserves those cues.
- `raw_stats` are compact summary features, while `raw_flattened` still carries richer temporal structure. LEM Brazil's raw-flattened advantage suggests some useful information is temporal or distributional beyond mean/min/max/quantiles.
- The result is still CropHarvest-only. It gates model design, but does not fix the second-benchmark gap.
- Calibrated F1 improves only modestly for the hybrid arm, so default-threshold F1 and AUROC should both remain in the paper tables.
- The table-grade rerun fixed the earlier convergence-warning issue; all probes converged with zero warnings.

### Next

The next model-design experiment should be a real raw-cue-preserving objective or architecture, but only after keeping this lesson narrow: preserve simple red-edge/NIR/NDVI/SAR cues without sacrificing JEPA transfer geometry. A good next run should avoid the old failed `[6]` path: use best-checkpoint selection, calibrated probes, raw/hybrid evaluation from the start, and no uncached teacher pass.

In parallel, keep second-benchmark ingestion high priority. `[6]` answers the raw-cue question on CropHarvest, but it does not establish that the hybrid/raw-cue story generalizes beyond this benchmark.

## [7] Temporal Block-JEPA v1

### Description

First serious architecture experiment for the JEPA branch. This is not a protocol-only rerun and not another CropHarvest pretraining run. It upgrades the current next-step prototype toward temporal masked latent prediction, pretrains on a bounded packaged SSL4EO-S12 v1.1 subset, then freezes the encoder and evaluates on strict CropHarvest probes.

The runner is `runners/[7].py`.

Dataset roles:

- Pretraining: bounded packaged SSL4EO-S12 v1.1 zarr, `N=49,152`, `T=4`, `H=W=16`. This uses exactly `768` source shards and a stream-and-evict cache capped at `60 GiB`. This is the first-screen replacement for the earlier `50k-100k` target. This is not yet a true agriculture-masked corpus.
- Evaluation/probes: CropHarvest v2 zarr only.

The pretrain and eval zarr stores must share the same channel contract:

- S2: `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`, `NDVI`
- S1: `VV`, `VH`
- Context: `temperature`, `precipitation`, `elevation`

Planned arms:

| Arm | Change |
|---|---|
| `A_control` | current causal next-embedding JEPA control |
| `B_full_target` | block model with full temporal EMA target encoder |
| `C_transformer_predictor` | adds mask-conditioned Transformer predictor |
| `D_multiblock` | adds temporal multi-block masking |
| `E_cross_modal` | adds cross-modal latent prediction/dropout |
| `F_rawcue` | adds raw-cue auxiliary target |
| `G_full` | full B+C+D+E+F candidate |

### Motivation

The current model tested only a modest causal next-embedding JEPA, not the full JEPA architecture family described in `notes/jepa.md`. `[6]` showed that simple raw red-edge/NIR/NDVI/SAR cues still improve strict probes, which means the representation is probably discarding useful agronomic magnitude information. `[7]` tests whether better JEPA mechanics plus patch-level external Sentinel pretraining can internalize those cues and improve embedding-only strict transfer.

### What We Hope To See

- Embedding-only full `[7]` approaches or beats the `[6]` hybrid.
- Strict priority embedding-only F1 reaches roughly `>= 0.53`.
- Strict priority AUROC reaches roughly `>= 0.69`.
- Calibrated F1 reaches roughly `>= 0.59`.
- LEM Brazil AUROC improves beyond `0.60`.
- The hybrid gap shrinks because embeddings preserve more raw agronomic cues.

### Results

Pulled artifacts:

- `artifacts/runs/temporal_block_jepa_v1_screen`
- aggregate priority table: `artifacts/runs/temporal_block_jepa_v1_screen/priority_summary.csv`
- per-heldout table: `artifacts/runs/temporal_block_jepa_v1_screen/per_holdout_priority_summary.csv`
- per-arm training histories: `artifacts/runs/temporal_block_jepa_v1_screen/*_seed42/train_history.json`

The full seven-arm screen completed successfully for seed `42`, all five strict heldouts, and all five stress conditions. This produced `2,625` probe-result rows. All probes converged with zero convergence warnings.

Strict priority aggregate:

| Arm | Embedding F1 | Embedding AUROC | Calibrated F1 | Hybrid F1 | Hybrid AUROC | Hybrid Calibrated F1 |
|---|---:|---:|---:|---:|---:|---:|
| `A_control` | 0.3657 | 0.6347 | 0.5508 | 0.4204 | 0.6741 | 0.5652 |
| `B_full_target` | **0.3711** | **0.6400** | **0.5514** | **0.4244** | **0.6798** | 0.5662 |
| `C_transformer_predictor` | 0.3591 | 0.6327 | 0.5454 | 0.4140 | 0.6746 | 0.5667 |
| `D_multiblock` | 0.3614 | 0.6366 | 0.5499 | 0.4159 | 0.6772 | 0.5665 |
| `E_cross_modal` | 0.3643 | 0.6358 | 0.5453 | 0.4186 | 0.6771 | **0.5671** |
| `F_rawcue` | 0.3519 | 0.6229 | 0.5416 | 0.4082 | 0.6659 | 0.5609 |
| `G_full` | 0.3557 | 0.6192 | 0.5405 | 0.4128 | 0.6639 | 0.5595 |
| raw stats only | 0.5086 | 0.6456 | 0.5617 | - | - | - |

Comparison against the repaired `[6]` CropHarvest-pretrained representation:

| Representation | Priority F1 | Priority AUROC | Calibrated F1 |
|---|---:|---:|---:|
| `[6]` `large_dual_s2 / surf_jepa_v0` | 0.4915 | 0.6655 | 0.5839 |
| `[7]` best embedding-only: `B_full_target` | 0.3711 | 0.6400 | 0.5514 |
| Difference | -0.1204 | -0.0255 | -0.0325 |
| `[6]` `large_dual_s2 / surf_jepa_v0_plus_raw_stats` | 0.5307 | 0.6939 | 0.5911 |
| `[7]` best hybrid: `B_full_target` | 0.4244 | 0.6798 | 0.5662 |
| Difference | -0.1063 | -0.0141 | -0.0249 |

Training loss reached its best validation value early and then degraded:

| Arm | Best Epoch | Best Validation Loss | Last Epoch |
|---|---:|---:|---:|
| `A_control` | 3 | 0.003975 | 8 |
| `B_full_target` | 5 | 0.000714 | 10 |
| `C_transformer_predictor` | 4 | 0.001247 | 9 |
| `D_multiblock` | 4 | 0.001263 | 9 |
| `E_cross_modal` | 4 | 0.001233 | 9 |
| `F_rawcue` | 5 | 0.012059 | 10 |
| `G_full` | 5 | 0.011710 | 10 |

Best-checkpoint reload was active, so the late loss drift wasted runtime but does not explain away the poor strict probes. The current `7e-5` schedule is still too aggressive for a clean next screen.

### Interpretation

`[7]` is a negative result. The packaged SSL4EO patch bridge does not improve the foundation model. Every `[7]` embedding arm loses substantially to the repaired CropHarvest-pretrained `[6]` encoder. Raw stats alone also beat every `[7]` embedding and hybrid arm by default F1.

The only useful architecture signal is small: `B_full_target` improves modestly over `A_control`, so a full temporal EMA target branch is worth retaining for the next controlled test. The Transformer predictor, temporal multi-block masking, cross-modal objective, and current raw-cue auxiliary loss did not produce gains. The raw-cue arms are among the worst arms, so the current auxiliary design should not be carried forward unchanged.

This result does not establish that temporal JEPA is better than Presto or OlmoEarth. It rejects a narrower hypothesis: generic packaged SSL4EO patches plus additional temporal JEPA mechanics are not enough to improve strict agricultural transfer.

### Issues

- `60 GiB` is a cache cap, not the amount of useful training data. Stream-and-evict preprocessing still produced `49,152` training samples. Raising the cap would not change the corpus.
- The packaged SSL4EO subset is generic global EO data, not an agriculture-filtered corpus. `NDVI_MEAN_MIN=None`, and no cropland mask is applied.
- SSL4EO-S12 supplies only four seasonal timesteps. This is not a long or irregular crop-sequence test.
- Exact acquisition dates were discarded by the local bridge builder even though the SSL4EO Zarr source exposes per-sample `time_` values. The local store currently replaces them with synthetic dates: March 15, June 15, September 15, and December 15. Repair ingestion before the next run.
- Climate is absent during SSL4EO pretraining. The local climate tensor is zero-filled and marked unavailable with `climate_mask=0`.
- Spatial information is compressed too early. `PatchEncoder` applies convolution and `AdaptiveAvgPool2d((1, 1))` independently at each timestep. The JEPA loss predicts temporal vectors after spatial pooling; it does not predict masked spatial-temporal tokens.
- The architecture is large relative to the effective signal: model dimension `768`, eight temporal layers, and only four temporal observations per sample.
- The learning-rate schedule is unstable after the early best epoch. A lower rate and shorter early-stopping patience are required for the next controlled screen, but optimization changes alone should not be expected to fix the representation mismatch.
- Pre-launch blockers fixed before launch: the EMA target branch is forced to eval mode, non-multiblock B/C masks use one short target block, validation masks are deterministic, `src/cropharvest.py` builds the v2 channel contract, `[7]` asserts exact band names/order, raw-cue targets reduce over spatial patches, and pretraining runs once per arm/seed before evaluating all holdouts.
- `crop_prior` is intentionally excluded from `[7]`. Add it only as a later ablation after building the same external crop-confidence prior for both SSL4EO pretraining samples and CropHarvest evaluation points.

### Next

Do not scale this recipe by downloading more generic SSL4EO shards. The next experiment must isolate whether the failure comes from domain mismatch or tokenization.

Build a small controlled bridge screen around `B_full_target`:

- lower the learning rate to `3e-5`,
- use early-stopping patience `2`,
- shorten the screen to `10-12` epochs,
- reduce the model to roughly `384` dimensions and `4` temporal layers,
- keep the strict CropHarvest probes,
- compare equal-sized generic SSL4EO and agriculture-filtered or vegetation-biased pretraining pools,
- retain the old CropHarvest-point pretraining path as a matched-domain control.

If agriculture-filtered pretraining materially improves transfer, build a custom crop-season chip corpus with exact dates and joined climate variables. If it does not, the next architecture must preserve spatial tokens through the encoder and predict spatial-temporal latent blocks instead of pooling each patch to one temporal vector before the JEPA loss.

## [8] Agriculture-Aligned Spatial-Token JEPA Bridge

The run used the official `embed2scale/SSL4EO-S12-v1.1` WebDataset release and the completed `[8]` preprocessing stores:

| Store | Samples | Purpose |
|---|---:|---|
| `data/processed/ssl4eo_s12_v11_generic_fixed_48k.zarr` | `49,152` | Generic external control |
| `data/processed/ssl4eo_s12_v11_agro_fixed_48k.zarr` | `49,152` | Agriculture-focused pool |

Preprocessing QA passed before training: schema and source revision matched, sample IDs were unique, pool overlap was `0`, cloud-mask parity passed on `100/100` checked samples, and generic/agriculture pools had identical geography and composite geography/clear-fraction/token-availability histograms.

Enabled arms:

| Arm | Data | Encoder | Masking |
|---|---|---|---|
| `A_pool_generic_fixed` | generic | corrected pooled reference | temporal |
| `B_pool_agro_fixed` | agriculture-focused | corrected pooled reference | temporal |
| `C_spatial_temporal` | agriculture-focused | spatial-token JEPA | temporal blocks |
| `D_spatiotemporal` | agriculture-focused | spatial-token JEPA | spatial-temporal multi-block |

The runner evaluated one seed (`42`), five strict heldouts, four degradation conditions, two robustness protocols, three probe families (`embedding`, `embedding_plus_raw_stats`, `raw_stats`), and both `best` and `final` checkpoints. The complete worker output contains `4,200` probe rows; root-level `probe_results.csv` contains the `2,100` final-checkpoint rows.

### Results

Strict priority final-checkpoint aggregate (`clean`, full label budget, clean-probe-to-clean-test):

| Arm | Probe | F1 | AUROC | Calibrated F1 |
|---|---|---:|---:|---:|
| `A_pool_generic_fixed` | embedding | **0.5972** | 0.7451 | 0.5692 |
| `A_pool_generic_fixed` | embedding + raw stats | 0.5830 | **0.7516** | 0.5960 |
| `B_pool_agro_fixed` | embedding | 0.5916 | 0.7314 | 0.5665 |
| `B_pool_agro_fixed` | embedding + raw stats | 0.5736 | 0.7371 | 0.5962 |
| `C_spatial_temporal` | embedding | 0.5398 | 0.7094 | **0.6089** |
| `C_spatial_temporal` | embedding + raw stats | 0.5440 | 0.7145 | 0.6077 |
| `D_spatiotemporal` | embedding | 0.5358 | 0.7156 | 0.6019 |
| `D_spatiotemporal` | embedding + raw stats | 0.5620 | 0.7240 | 0.6048 |
| raw stats only | raw stats | 0.5282 | 0.6952 | 0.5844 |

Per-heldout final priority for the best default-F1 arm (`A_pool_generic_fixed`, embedding):

| Heldout | F1 | AUROC | Calibrated F1 |
|---|---:|---:|---:|
| `ethiopia` | 0.8012 | 0.8513 | 0.7471 |
| `lem-brazil` | 0.3577 | 0.6455 | 0.3150 |
| `rwanda-ceo` | 0.3991 | 0.6150 | 0.5472 |
| `togo` | 0.7660 | 0.8021 | 0.7167 |
| `togo-eval` | 0.6620 | 0.8115 | 0.5198 |

Best-checkpoint versus final-checkpoint priority for embedding-only probes:

| Arm | Best F1 | Final F1 | Best AUROC | Final AUROC | Best Calibrated F1 | Final Calibrated F1 |
|---|---:|---:|---:|---:|---:|---:|
| `A_pool_generic_fixed` | **0.5998** | 0.5972 | 0.7103 | **0.7451** | **0.5740** | 0.5692 |
| `B_pool_agro_fixed` | 0.5900 | **0.5916** | 0.7115 | **0.7314** | **0.5815** | 0.5665 |
| `C_spatial_temporal` | **0.5796** | 0.5398 | **0.7185** | 0.7094 | 0.5983 | **0.6089** |
| `D_spatiotemporal` | **0.5959** | 0.5358 | **0.7330** | 0.7156 | 0.5998 | **0.6019** |

Training validation loss reached its minimum very early, then increased and flattened at a worse value:

| Arm | Best Validation Epoch | Best Validation Loss | Final Validation Loss | Train Time |
|---|---:|---:|---:|---:|
| `A_pool_generic_fixed` | 1 | 0.002916 | 0.033661 | 16.0 min |
| `B_pool_agro_fixed` | 1 | 0.001032 | 0.074249 | 16.0 min |
| `C_spatial_temporal` | 2 | 0.004645 | 0.025566 | 15.2 min |
| `D_spatiotemporal` | 2 | 0.005113 | 0.020005 | 20.9 min |

Final-checkpoint robustness for embedding-only probes under the primary `clean_train_degraded_test` protocol:

| Arm | Clean F1 / AUROC | S2-off F1 / AUROC | Temporal-drop-50 F1 / AUROC | S2-off + temporal-drop-50 F1 / AUROC |
|---|---:|---:|---:|---:|
| `A_pool_generic_fixed` | 0.597 / 0.745 | 0.102 / 0.702 | 0.571 / 0.726 | 0.118 / 0.688 |
| `B_pool_agro_fixed` | 0.592 / 0.731 | 0.521 / 0.489 | 0.575 / 0.704 | 0.506 / 0.484 |
| `C_spatial_temporal` | 0.540 / 0.709 | 0.139 / 0.575 | 0.520 / 0.651 | 0.141 / 0.569 |
| `D_spatiotemporal` | 0.536 / 0.716 | 0.552 / 0.587 | 0.461 / 0.567 | 0.552 / 0.597 |

When probes are retrained under the same degraded condition (`condition_matched_retrained`), all arms recover to roughly `0.54-0.58` F1 under S2-off and S2-off-plus-temporal-drop. That means the degraded embeddings retain some usable signal, but the clean-trained probe does not transfer cleanly into the degraded feature distribution.

### Interpretation

`[8]` is a positive clean-transfer result and a negative robustness result.

The positive part: the external SSL4EO bridge now works under clean strict transfer. The best final embedding-only arm reaches `0.5972` F1 / `0.7451` AUROC, beating raw stats alone (`0.5282` / `0.6952`) and beating the repaired `[6]` CropHarvest-pretrained embedding-only result (`0.4915` / `0.6655`). The run also beats the repaired `[6]` hybrid result on default F1 and AUROC (`0.5307` / `0.6939`) without needing raw stats.

The negative part: `[8]` does not yet give a defensible robustness claim. `A_pool_generic_fixed` has the best clean F1/AUROC but collapses in default F1 when S2 is removed under clean-probe-to-degraded-test. `B_pool_agro_fixed` and `D_spatiotemporal` keep default F1 under S2-off, but their AUROC drops close to random or weak ranking. That is not robust representation learning; it is a warning that thresholds and feature distributions are shifting under degradation.

The agriculture-filtered pool did not help. `B_pool_agro_fixed` is slightly worse than the generic control `A_pool_generic_fixed` on clean F1 and AUROC. The current LULC-driven agriculture sampling may be reducing useful diversity, or it may be selecting visually agricultural patches without producing the temporal/crop-state signal CropHarvest needs.

The spatial-token path is not yet better than the pooled path. `C_spatial_temporal` and `D_spatiotemporal` trail the pooled arms on default clean F1 and AUROC. However, spatial arms have the best calibrated F1, and the best checkpoint for `D_spatiotemporal` reaches `0.5959` F1 / `0.7330` AUROC, close to the pooled generic arm. The spatial idea is not dead, but the final checkpoint and clean robustness behavior are not acceptable yet.

The JEPA validation loss is not a reliable downstream checkpoint selector. Validation loss is best at epoch `1-2`, but final checkpoints often improve AUROC while worsening loss. Future screens need a source-domain downstream proxy or a fixed checkpoint policy; JEPA validation loss alone should not decide the checkpoint.

### Next

Do not claim robustness from `[8]`. The claim available from this run is narrower and useful: external SSL4EO pretraining with corrected dates, normalization, and pool QA can produce better clean strict-transfer embeddings than the prior CropHarvest-only runs.

The next experiment should isolate the degradation failure:

- keep `A_pool_generic_fixed`, `B_pool_agro_fixed`, and `D_spatiotemporal`;
- add explicit modality-drop training rather than only degradation-time stress;
- evaluate whether S2-off collapse is a threshold/calibration problem or a real ranking problem by reporting both clean-probe-degraded-test and condition-matched-retrained results as first-class outputs;
- use a checkpoint selector that is not raw JEPA validation loss;
- repeat the best arm across multiple seeds before treating the `0.5972` clean result as stable.

## [9] Robust View-JEPA Modality-Drop Alignment

### Core Question

Can we keep `[8]`'s clean strict-transfer gain while making the embedding space stable when Sentinel-2 or timesteps are missing? `[8]` answered that corrected external SSL4EO pretraining can work for clean strict transfer. `[9]` should test whether the representation can stay useful under the missing-sensor and sparse-time conditions we care about. The main change is to move robustness from evaluation-only stress into the pretraining objective:

| Experiment | Context/Online View | EMA Target View | Robustness Exposure |
|---|---|---|---|
| `[8]` | clean input | clean input | tested only after training |
| `[9]` | clean or degraded input | clean/full input | trained directly |

In `[9]`, degraded context views should include `S2-off`, temporal-drop, and `S2-off + temporal-drop`. The EMA target should remain the clean/ full view. The model is trained to predict or align clean latent targets from degraded observations.

### Proposed Arms

| Arm | Data | Encoder | Training View | Extra Loss | Purpose |
|---|---|---|---|---|---|
| `A_lr_control_generic` | generic 48k | pooled | clean only | JEPA | Checks whether lower LR and fixed-checkpoint policy alone explain `[8]` |
| `B_generic_viewdrop` | generic 48k | pooled | degraded context, clean target | JEPA | Tests whether training-time dropout fixes S2-off failure |
| `C_mixed_viewdrop` | generic + agro 98k | pooled | degraded context, clean target | JEPA | Tests whether agro data helps as a mixture rather than a replacement |
| `D_mixed_viewdrop_consistency` | generic + agro 98k | pooled | clean/degraded paired views | JEPA + embedding consistency | Directly attacks clean/degraded feature drift |
| `E_spatial_viewdrop_consistency` | generic + agro 98k | spatial-token | clean/degraded paired views | JEPA + embedding consistency | Tests the final-shape spatial architecture under the right robustness objective |

The runner reuses the completed `[8]` preprocessing stores and does not require new preprocessing.

### Differences From `[8]`

| Dimension | `[8]` | `[9]` |
|---|---|---|
| Main question | Does corrected SSL4EO external pretraining improve clean transfer? | Can the representation stay stable under missing sensors/timesteps? |
| Training input | Mostly clean views | Clean and degraded paired views |
| Robustness | Evaluation-only stress | Training objective includes stress |
| Data | Generic and pure-agriculture pools tested separately | Generic control plus mixed generic/agriculture pool |
| Checkpointing | Best/final checkpoints | Fixed checkpoint sweep: epochs `1`, `2`, `4`, `8`, `12` |
| LR | `3e-5`, unstable loss curve | `1e-5` or `1.5e-5` |
| Loss interpretation | JEPA validation loss reported but not reliable | JEPA validation loss treated as weak diagnostic; downstream checkpoint sweep is primary |
| Architecture | Pooled and spatial screen | Pooled stable baseline; spatial only under robust view-drop objective |

### Changes Based On Failures Or Bugs

- Lower LR and fixed checkpoint sweep.
  - Based on `[8]`.
  - `[8]` validation loss reached its minimum at epoch `1-2`, then increased and flattened at a worse value.
  - Use `1e-5` or `1.5e-5`, and evaluate fixed checkpoints instead of trusting JEPA validation loss.

- Train-time modality and temporal dropout.
  - Based on `[8]`.
  - `[8]` improved clean transfer but did not support a robustness claim.
  - `A_pool_generic_fixed` reached clean F1/AUROC `0.5972/0.7451`, but S2-off default F1 collapsed to `0.1018` under clean-probe-to-degraded-test.
  - The model needs to see degraded context views during pretraining.

- Clean/degraded feature alignment.
  - Based on `[8]`.
  - Condition-matched retraining recovered S2-off performance to roughly `0.54-0.58` F1.
  - That implies degraded embeddings still contain signal, but clean-trained probes do not transfer across the feature shift.
  - Add a consistency loss aligning clean and degraded embeddings for the same sample.

- Mixed generic/agriculture data instead of pure agriculture replacement.
  - Based on `[8]`.
  - `B_pool_agro_fixed` was slightly worse than `A_pool_generic_fixed` on clean F1 and AUROC.
  - Pure LULC-filtered agriculture sampling may reduce useful diversity.
  - Test generic+agriculture mixture rather than replacing generic with agriculture-only data.

- Keep raw stats as diagnostics, not the primary training target.
  - Based on `[6]` and `[7]`.
  - Raw and hybrid probes showed agronomic magnitude cues matter, but explicit raw-cue auxiliary training did not become the winning path.
  - Continue reporting raw stats and hybrid probes, but do not center `[9]` on the old raw-cue auxiliary design.

- Do not scale the `[7]` recipe.
  - Based on `[7]`.
  - `[7]` showed that generic external patches plus additional JEPA mechanics were not enough when the data bridge was flawed.
  - `[8]` fixed the data path and produced the clean-transfer jump, so `[9]` should build from `[8]`.

### New Data, Features, And Architecture Changes

- Mixed generic+agriculture SSL pool.
  - This is closer to the final desired model than either pure generic or pure agriculture data.
  - The model should learn crop-relevant structure without losing broader EO diversity.

- Paired clean/degraded SSL views.
  - For each batch, generate clean, S2-off, temporal-drop, and S2-off-plus-temporal-drop views.
  - The online/context encoder sees degraded views; the EMA target sees clean views.

- Clean/degraded embedding consistency.
  - Add a global representation loss that explicitly reduces feature drift between clean and degraded views of the same sample.
  - This directly targets the failure mode exposed by `[8]`.

- Missingness-aware embeddings.
  - Add learned embeddings for sensor state and observation state: S2 present, S2 absent, S1 present, timestep observed, timestep dropped.
  - Current models receive masks, but they do not necessarily learn a stable missingness-conditioned embedding geometry.

- Spatial model only under robust objective.
  - `[8]` spatial-token arms did not beat pooled arms under clean training.
  - The best checkpoint for `D_spatiotemporal` still reached `0.5959` F1 / `0.7330` AUROC, so the spatial idea remains viable.
  - Retest spatial tokens only with view-drop and consistency, because that is closer to the final desired architecture.

### Success Criteria

Minimum useful result:

- Clean F1 remains near `[8]`: `>= 0.58`.
- Clean AUROC remains near `[8]`: `>= 0.73`.
- S2-off AUROC improves materially: `>= 0.65`.
- S2-off F1 does not collapse: `>= 0.50`.
- S2-off-plus-temporal-drop behaves similarly to S2-off alone.
- The gap between clean-probe-degraded-test and condition-matched-retrained results narrows.

Strong result:

- `D_mixed_viewdrop_consistency` or `E_spatial_viewdrop_consistency` reaches clean F1 around `0.60` and S2-off AUROC above `0.68`.
- This would move the project from a clean-transfer story toward a defensible robustness story.

### Deferred

Do not add climate in `[9]`. Climate is important for the final model, but `[8]` exposed a core missing-modality alignment problem. Fix that first; otherwise climate becomes another confound.

### Run Coverage

The completed run evaluated all five proposed arms with seed `42`, five strict geographic heldouts, four degradation conditions, both robustness protocols, and fixed checkpoints at epochs `1`, `2`, `4`, `8`, and `12`.

| Item | Value |
|---|---:|
| Training runs | `5` |
| Consolidated probe rows | `13,125` |
| Holdouts | `5` |
| Checkpoints per arm | `5` |
| Generic-pool training time per arm | `15.9` min |
| Mixed-pool training time per pooled arm | `29.6-29.8` min |
| Mixed-pool spatial-arm training time | `44.5` min |

### Results

Final-checkpoint embedding-only results under the primary `clean_train_degraded_test` protocol:

| Arm | Clean F1 / AUROC | S2-off F1 / AUROC | Temporal-drop-50 F1 / AUROC | S2-off + temporal-drop-50 F1 / AUROC |
|---|---:|---:|---:|---:|
| `A_lr_control_generic` | 0.588 / 0.738 | 0.005 / 0.700 | 0.559 / 0.715 | 0.005 / 0.679 |
| `B_generic_viewdrop` | 0.591 / 0.746 | 0.000 / 0.658 | 0.561 / 0.719 | 0.000 / 0.647 |
| `C_mixed_viewdrop` | 0.584 / 0.743 | **0.572 / 0.702** | 0.569 / 0.717 | **0.570 / 0.684** |
| `D_mixed_viewdrop_consistency` | 0.590 / 0.750 | 0.284 / **0.713** | 0.568 / **0.723** | 0.290 / 0.682 |
| `E_spatial_viewdrop_consistency` | **0.601 / 0.756** | 0.566 / 0.697 | **0.573** / 0.693 | 0.559 / 0.644 |

`C_mixed_viewdrop` is the strongest robustness result. It gives up little clean performance relative to `[8]` while eliminating the catastrophic default-F1 failure when S2 is absent:

| Model | Clean F1 / AUROC | S2-off F1 / AUROC | S2-off + temporal-drop-50 F1 / AUROC |
|---|---:|---:|---:|
| `[8] A_pool_generic_fixed` | 0.597 / 0.745 | 0.102 / 0.702 | 0.118 / 0.688 |
| `[9] C_mixed_viewdrop` | 0.584 / 0.743 | **0.572 / 0.702** | **0.570 / 0.684** |
| Difference | -0.013 / -0.002 | **+0.470 / +0.000** | **+0.452 / -0.004** |

This comparison is important: `[9] C` fixes the clean-probe-to-degraded-test default-F1 collapse without improving degraded AUROC. The degraded representation already retained ranking signal in `[8]`; `[9]` primarily makes the clean and degraded feature distributions compatible enough for the same probe and threshold to work.

The condition-matched comparison confirms that conclusion:

| Arm | S2-off Clean-Probe F1 / AUROC | S2-off Condition-Matched F1 / AUROC | F1 Gap |
|---|---:|---:|---:|
| `A_lr_control_generic` | 0.005 / 0.700 | 0.582 / 0.721 | -0.577 |
| `B_generic_viewdrop` | 0.000 / 0.658 | 0.577 / 0.722 | -0.577 |
| `C_mixed_viewdrop` | **0.572 / 0.702** | 0.571 / 0.724 | **+0.000** |
| `D_mixed_viewdrop_consistency` | 0.284 / 0.713 | 0.574 / 0.729 | -0.290 |
| `E_spatial_viewdrop_consistency` | 0.566 / 0.697 | 0.575 / 0.726 | -0.008 |

`C` and `E` nearly eliminate the F1 gap between a clean-trained probe and a condition-matched probe. `D` improves S2-off ranking but still leaves a large default-threshold shift.

Final-checkpoint clean embedding results:

| Arm | F1 | AUROC | Calibrated F1 |
|---|---:|---:|---:|
| `A_lr_control_generic` | 0.5878 | 0.7382 | 0.5682 |
| `B_generic_viewdrop` | 0.5914 | 0.7464 | 0.5703 |
| `C_mixed_viewdrop` | 0.5839 | 0.7426 | 0.5651 |
| `D_mixed_viewdrop_consistency` | 0.5897 | 0.7495 | 0.5737 |
| `E_spatial_viewdrop_consistency` | **0.6013** | **0.7562** | **0.5802** |
| Raw stats only | 0.5282 | 0.6952 | 0.5844 |

Before the verification pass, `E_spatial_viewdrop_consistency` appeared to be the strongest clean final-checkpoint model and the first spatial-token arm to beat the pooled reference cleanly. It exceeds `[8] A_pool_generic_fixed` on clean F1 and AUROC while retaining usable default F1 under S2-off. Its combined S2-off-plus-temporal-drop AUROC, however, falls to `0.644`, materially below `C`'s `0.684`. The native verification below shows why this downstream result is not enough to make current `E` the next scale target.

Best clean-F1 checkpoint per arm:

| Arm | Checkpoint | F1 | AUROC |
|---|---:|---:|---:|
| `A_lr_control_generic` | epoch `4` | 0.6066 | 0.7281 |
| `B_generic_viewdrop` | epoch `4` | **0.6142** | 0.7360 |
| `C_mixed_viewdrop` | epoch `4` | 0.6048 | **0.7523** |
| `D_mixed_viewdrop_consistency` | epoch `2` | 0.5990 | 0.7215 |
| `E_spatial_viewdrop_consistency` | epoch `12` | 0.6013 | 0.7562 |

The fixed checkpoint sweep was necessary. Pooled-arm clean F1 peaks at epoch `2-4`, while pooled-arm AUROC generally improves later. `E` is different: its validation loss decreases throughout training and its final checkpoint is its strongest clean checkpoint. Verification later shows that this low-loss behavior is suspicious rather than automatically desirable.

Training-loss behavior:

| Arm | Best Validation Epoch | Best Validation Loss | Final Validation Loss |
|---|---:|---:|---:|
| `A_lr_control_generic` | 1 | 0.003199 | 0.040067 |
| `B_generic_viewdrop` | 1 | 0.003467 | 0.035651 |
| `C_mixed_viewdrop` | 1 | 0.001894 | 0.047063 |
| `D_mixed_viewdrop_consistency` | 2 | 0.000502 | 0.002625 |
| `E_spatial_viewdrop_consistency` | **12** | **0.000654** | **0.000654** |

Lowering the learning rate did not fix the pooled JEPA loss pattern. `A`, `B`, and `C` still reach minimum validation loss at epoch `1`; `D` reaches it at epoch `2`. Only the spatial consistency arm trains monotonically through the full schedule. JEPA validation loss remains unsuitable as the sole downstream checkpoint selector.

### Deeper Loss Diagnosis

The epoch-`1` pooled validation minima should not currently be interpreted as ordinary overfitting or proof that later training is wasted.

The epoch-`1` training loss averages batches from the entire first epoch, including the initially random predictor and the substantially improved predictor near the end. Validation occurs only after that full epoch. A validation loss below the epoch-average training loss is therefore expected and is not itself evidence of leakage or a broken split.

More importantly, JEPA loss is nonstationary because the EMA target encoder changes throughout training. For the pooled arms, higher validation loss is strongly associated with better clean downstream AUROC:

| Arm | Correlation: Validation Loss vs Clean AUROC |
|---|---:|
| `A_lr_control_generic` | `+0.973` |
| `B_generic_viewdrop` | `+0.976` |
| `C_mixed_viewdrop` | `+0.954` |
| `D_mixed_viewdrop_consistency` | `+0.810` |
| `E_spatial_viewdrop_consistency` | `-0.834` |

The likely pooled-model failure mode at the early minimum is an easy positional shortcut:

- SSL4EO provides only four seasonal timesteps.
- The pooled target mask hides two timesteps and never hides timestep `0`.
- The predictor receives mask tokens plus strong day-of-year, elapsed-time, and position identity.
- The EMA target initially contains weak, low-diversity representations that can be predicted largely from temporal identity.
- As the target encoder becomes more content-sensitive, the prediction problem becomes harder and JEPA loss rises even while downstream representation quality improves.

Preliminary checkpoint diagnostics on point-timeseries inputs supported this interpretation:

| Arm | Checkpoint | Same-Time Inter-Sample Cosine | Effective Rank | Shuffled-Target Loss |
|---|---:|---:|---:|---:|
| `A_lr_control_generic` | epoch `1` | `0.989` | `1.66` | `0.00795` |
| `A_lr_control_generic` | epoch `12` | `0.657` | `2.46` | `0.405` |
| `C_mixed_viewdrop` | epoch `1` | `0.992` | `1.58` | `0.0058` |
| `C_mixed_viewdrop` | epoch `12` | `0.612` | `3.36` | `0.476` |
| `D_mixed_viewdrop_consistency` | epoch `2` | `0.998` | not recorded | `0.00136` |
| `D_mixed_viewdrop_consistency` | epoch `12` | `0.968` | `2.71` | `0.0366` |

`A` and `C` became substantially more sample-specific and content-sensitive despite their rising JEPA loss. `D` remained much less diverse, which is consistent with the global consistency objective suppressing useful content variation.

`E` required separate scrutiny. Its monotonically falling loss looked conventionally healthy, but the same point-input diagnostic found extremely high same-time inter-sample cosine and weak shuffled-target separation. That concern was tested directly in the native SSL4EO verification run below.

Consequences:

- Do not early-stop pooled JEPA using raw validation loss.
- Do not treat the epoch-`1` pooled checkpoint as the best representation.
- Do not treat `E`'s near-zero loss as proof of superior representation learning.
- Future runs must measure content sensitivity and representation diversity directly on a fixed native SSL4EO validation set.

### Native SSL4EO Verification

After `[9]`, `src/temp/verify_9.py` evaluated all saved checkpoints on two native SSL4EO validation pools:

| Pool | Samples | Meaning |
|---|---:|---|
| `generic_common_heldout` | `655` | generic samples held out from both generic-only and mixed `[9]` training |
| `agriculture_heldout` | `2048` | agriculture samples held out from mixed `[9]` training |

Artifacts:

- `artifacts/[9]/verification/checkpoint_diagnostics.csv`
- `artifacts/[9]/verification/corruption_diagnostics.csv`
- `artifacts/[9]/verification/covariance_spectra.json`
- `artifacts/[9]/verification/validation_sets.json`

The verification grid is complete:

| File | Rows | Expected |
|---|---:|---:|
| `checkpoint_diagnostics.csv` | `50` | `2 pools * 5 arms * 5 checkpoints` |
| `corruption_diagnostics.csv` | `200` | `2 pools * 5 arms * 5 checkpoints * 4 corruptions` |
| `covariance_spectra.json` | `50` entries | `2 pools * 5 arms * 5 checkpoints` |

The `corruption` rows are intentional synthetic stress diagnostics, not damaged data. They evaluate:

- `sensor_off_s2`;
- `sensor_off_s1`;
- `temporal_drop_50`;
- `s2_off_tdrop50`.

Final-checkpoint content diagnostics:

| Pool | Arm | Local Loss | Content- Shuffle Gap | Target Rank | Same-Time Cosine | Clean-Zero Cosine |
|---|---|---:|---:|---:|---:|---:|
| `generic_common_heldout` | `A_lr_control_generic` | `0.0336` | `0.3371` | `2.11` | `0.604` | `-0.249` |
| `generic_common_heldout` | `B_generic_viewdrop` | `0.0261` | `0.2626` | `1.97` | `0.685` | `-0.175` |
| `generic_common_heldout` | `C_mixed_viewdrop` | `0.0284` | `0.3116` | `3.04` | `0.612` | `-0.238` |
| `generic_common_heldout` | `D_mixed_viewdrop_consistency` | `0.0018` | `0.0190` | `2.40` | `0.975` | `0.519` |
| `generic_common_heldout` | `E_spatial_viewdrop_consistency` | `0.0005` | `0.0019` | `2.24` | `0.998` | `0.993` |
| `agriculture_heldout` | `A_lr_control_generic` | `0.0415` | `0.1451` | `1.77` | `0.797` | `-0.377` |
| `agriculture_heldout` | `B_generic_viewdrop` | `0.0316` | `0.1073` | `1.63` | `0.845` | `-0.281` |
| `agriculture_heldout` | `C_mixed_viewdrop` | `0.0347` | `0.1422` | `2.64` | `0.781` | `-0.413` |
| `agriculture_heldout` | `D_mixed_viewdrop_consistency` | `0.0018` | `0.0058` | `2.00` | `0.990` | `0.492` |
| `agriculture_heldout` | `E_spatial_viewdrop_consistency` | `0.0004` | `0.0005` | `2.47` | `0.999` | `0.994` |

This changes the interpretation of `[9]`:

- `C_mixed_viewdrop` is the best trustworthy candidate. It has strong content-shuffle separation, the best or near-best effective rank, and good downstream robustness.
- Current `D_mixed_viewdrop_consistency` is not a good objective. It drives JEPA loss down while sharply reducing content sensitivity.
- Current `E_spatial_viewdrop_consistency` should not be scaled. Its near-zero loss is paired with near-zero content-shuffle gap, near-identical clean-zero target representations, and almost identical same-time embeddings across samples. The spatial architecture may still be useful, but the current spatial consistency recipe is solving too much of the task through low-content structure.
- `sensor_off_s1` reveals a broader missing-sensor weakness. Pooled `A/B/C` have very large displacement when S1 is removed. If the project claims sensor-dropout robustness rather than only cloud/S2 robustness, S1-off must become a training-time corruption.

### Interpretation

`[9]` is a positive result, but the exact positive claim matters.

The experiment succeeds at robust feature alignment. `C_mixed_viewdrop` preserves almost all of `[8]`'s clean AUROC while moving S2-off default F1 from `0.102` to `0.572` and combined-drop default F1 from `0.118` to `0.570`. Its clean-trained degraded-condition F1 is effectively identical to condition-matched retraining. That is direct evidence that the same downstream decision rule transfers across clean and degraded embeddings.

The experiment does not establish materially better degraded-condition ranking. `C`'s S2-off and combined-drop AUROCs are effectively unchanged from `[8]`. The model has learned a substantially more compatible embedding geometry, but it has not extracted more task information from the remaining sensors. Future work must distinguish representation alignment from actual missing-modality inference.

Generic-pool view-drop alone is insufficient. `B_generic_viewdrop` still collapses to zero default F1 under S2-off at the final checkpoint. The success appears only when view-drop is paired with the mixed generic/agriculture pool in `C`, but this comparison is confounded: `C` has twice as many unique samples and approximately twice as many optimizer steps per epoch as `B`. `[9]` therefore cannot determine whether the gain comes from agricultural mixture, greater diversity, or greater training exposure.

The explicit global consistency loss is not the winning pooled design. Relative to `C`, `D` improves final clean and S2-off AUROC, but it reintroduces severe default-F1 shift under S2-off. This suggests that global cosine alignment preserves ranking without sufficiently stabilizing the probe boundary. The current consistency loss should not automatically become part of the pooled default.

The spatial-token architecture remains viable, but the current spatial consistency recipe is not trustworthy. `E` is the strongest clean final model and nearly eliminates the S2-off clean-probe versus condition-matched F1 gap, but the native verification shows that its low JEPA loss is paired with almost no content-shuffle separation. The result should be treated as a warning: spatial tokens can look good downstream while the SSL objective becomes too easy.

The persistent geographic failures remain. Even the strongest arms remain weak on `lem-brazil` and `rwanda-ceo`, particularly under missing S2. Robust feature alignment does not solve the existing cross-region and class-imbalance failures.

The predeclared strong criterion was numerically met by `E`: clean F1 is approximately `0.60`, and S2-off AUROC is above `0.68`. Verification weakens that result substantially. Do not use the current `E` recipe as the next scale target.

### Issues

- Only seed `42` was run.
- `B` versus `C` confounds mixed-data composition, unique sample count, and optimizer-step count.
- `[9]` did not isolate learned missingness-state embeddings as their own arm.
- The consistency objective is evaluated only at one weight, `0.25`.
- Default F1, calibrated F1, and AUROC still disagree sharply for several arms and checkpoints. Report all three; none is sufficient alone.
- Lower learning rate did not repair pooled JEPA validation-loss drift.
- `[9]` did not log content diagnostics during training; they were only added in a post-hoc verification script.
- The current global consistency loss suppresses content diversity.
- The apparently healthy spatial loss curve is compatible with a positional/content-insensitive shortcut.
- S1-off was not used as a training-time view-drop mode, but verification shows it is a major stress condition.

### Next

Do not rerun `[9]` unchanged, and do not scale current `E`.

The next training experiment should:

- keep `C_mixed_viewdrop` as the reference recipe;
- remove the data-size and optimizer-step confound between `B` and `C`;
- add S1-off as a training-time view-drop mode if claiming sensor-dropout robustness;
- remove current global consistency from spatial arms;
- test spatial content regularization instead of global clean/degraded cosine consistency;
- log the verification diagnostics during training rather than only after training;
- add EuroCropsML as an external evaluation target, not as new SSL pretraining data.

## [10] Matched-Control Content-Sensitive JEPA With External Transfer

### Core Question

Which parts of `[9]` genuinely improve learned representations, and can the selected representation transfer beyond CropHarvest?

`[10]` should separate agricultural data composition from optimizer exposure, replace the failing spatial consistency objective with content-preserving spatial variants, add S1-off to the robustness training distribution, and evaluate the resulting representations on EuroCropsML under its official cross-country few-shot protocol.

This is not a scale-up experiment. It is the controlled experiment required before choosing the architecture and data recipe to scale.

### Changes Based On Failures Or Ambiguities In Earlier Experiments

| Change | Source | Problem Addressed |
|---|---|---|
| Train every arm for the same number of optimizer updates | `[9]` | `B` and `C` differed in data composition, unique samples, and update count |
| Evaluate checkpoints by update count instead of epoch | `[9]` | Epochs represented different amounts of optimization across pools |
| Log content-sensitivity and representation-diversity diagnostics during training | `[9]` verification | Raw JEPA validation loss rewards low-content spatial shortcuts |
| Remove current global consistency from the main spatial candidate | `[9]` verification | `D` and `E` reduce loss while suppressing content sensitivity |
| Add content-diversity regularization to one spatial arm | `[9]` verification | Spatial tokens need anti-collapse pressure, not stronger invariance |
| Add S1-off view-drop modes | `[9]` verification | Pooled `A/B/C` are highly sensitive to removing S1 |
| Repeat reference candidates across seeds `42`, `43`, and `44` | `[9]` | All current conclusions depend on one seed |
| Preserve all clean/degraded probe protocols | `[8]`, `[9]` | Alignment gains must remain distinguishable from ranking gains |

Use fixed optimizer updates instead of epochs. The target budget should match the approximate update count of `[9] C`: `16,600` optimizer steps per seed per arm. Save and evaluate checkpoints at fixed update counts:

| Checkpoint | Purpose |
|---:|---|
| `1,000` | early shortcut phase |
| `3,000` | early content transition |
| `6,000` | middle training |
| `10,000` | late training |
| `16,600` | final |

### New Data, Evaluation, And Architecture Changes

| Addition | Purpose |
|---|---|
| EuroCropsML external evaluation | Tests whether the representation transfers to different countries, labels, preprocessing, and temporal structure |
| Official Latvia/Portugal-to-Estonia few-shot protocol | Tests cross-country and sparse-label transfer without inventing a favorable split |
| Actual irregular timestamps and long variable-length sequences | Tests temporal reasoning that four-season SSL4EO inputs cannot establish |
| S1-off training corruptions | Expands robustness beyond cloud/S2 dropout |
| Spatial arm without global consistency | Tests whether spatial tokens help when the failing consistency loss is removed |
| Spatial arm with variance/covariance content regularization | Prevents low-diversity or position-only solutions while preserving spatial information |
| UMAP by crop class and seasonal phase | Begins embedding interpretation using external multiclass labels |
| Phenological residual probe | Tests prediction of future vegetation change beyond current spectral state and day-of-year |

Do not add climate to `[10]`. Climate reintegration remains important, but adding it here would confound the objective-integrity and external-transfer questions. Climate should be introduced after `[10]` selects a trustworthy architecture and training objective.

EuroCropsML is S2-only and parcel-level. It can evaluate temporal representation quality, multiclass crop transfer, sparse labels, and temporal dropout. It cannot evaluate S2-off robustness or prove that spatial tokens help. Spatial models evaluated on EuroCropsML must be described as using their temporal/global representation head.

### Training Corruptions

Every view-drop arm should sample context modes uniformly from:

| Mode | Context View | Target View |
|---|---|---|
| `clean` | full input | clean/full input |
| `sensor_off_s2` | Sentinel-2 removed | clean/full input |
| `sensor_off_s1` | Sentinel-1 removed | clean/full input |
| `temporal_drop_50` | 50% timesteps removed, timestep `0` retained | clean/full input |
| `s2_off_tdrop50` | Sentinel-2 removed and 50% timesteps removed | clean/full input |
| `s1_off_tdrop50` | Sentinel-1 removed and 50% timesteps removed | clean/full input |

The target branch remains clean/full. Temporal-drop losses and embedding-consistency diagnostics must exclude dropped timesteps from the context branch, but the full-clean versus degraded displacement should still be logged for evaluation diagnostics.

### Target Masking

Use the `[9]` mask style for the pooled matched-control arms so the `B/C/A` comparisons isolate data and update-count effects. For spatial arms, remove the hard-coded assumption that timestep `0` is never a target. Spatial target masks should satisfy:

- at least one timestep remains visible as context;
- target cells are contiguous blocks;
- target budget remains exactly `50%`;
- masks can include timestep `0`;
- the mask generator logs per-timestep target counts.

This directly attacks the positional-shortcut pathway found in `[9]`.

### Proposed Arms

| Arm | Data | Encoder | Objective | Purpose |
|---|---|---|---|---|
| `A_mixed_viewdrop_reference` | full mixed pool, `98k` | pooled | `[9] C` view-drop JEPA, no S1-off | Reproduce `[9] C` under fixed-update, multi-seed conditions |
| `B_mixed_viewdrop_sensor_dropout` | full mixed pool, `98k` | pooled | view-drop JEPA with S1-off modes added | Tests whether adding S1-off improves broad sensor robustness without hurting S2 robustness |
| `C_generic_step_matched` | generic `48k`, repeated/resampled | pooled | same as `B` | Tests whether mixed-pool gains are just update exposure |
| `D_mixed_size_matched` | deterministic `24k` generic + `24k` agriculture | pooled | same as `B` | Tests agriculture mixture at fixed sample count and update count |
| `E_spatial_no_consistency` | full mixed pool, `98k` | spatial | view-drop JEPA with S1-off modes, no global consistency | Tests the spatial architecture without the failing consistency objective |
| `F_spatial_content_regularized` | full mixed pool, `98k` | spatial | `E` plus variance/covariance regularization | Tests a content-sensitive spatial final-shape candidate |

Run seeds `42`, `43`, and `44` for all arms unless runtime becomes prohibitive. If runtime must be cut, keep three seeds for `A`, `B`, `E`, and `F`, and run `C/D` at seed `42` as diagnostic controls. The preferred run is all arms across all three seeds.

### Content Regularization

The content regularizer should protect representation variance and covariance structure. It must not be another clean/degraded cosine-consistency term.

Apply the regularizer to context/global embeddings from both clean and degraded views:

- compute sequence embeddings using the same pooling path as downstream extraction;
- for temporal-drop views, compute one embedding over retained timesteps and one full-clean embedding for displacement diagnostics;
- variance term: penalize dimensions whose batch standard deviation falls below a floor;
- covariance term: penalize large off-diagonal covariance after centering;
- apply to clean and degraded embeddings separately;
- log the regularizer terms separately from JEPA loss.

Suggested starting weights:

| Term | Weight |
|---|---:|
| variance floor loss | `1.0` |
| off-diagonal covariance loss | `0.04` |
| total content regularizer multiplier | `0.05` |

These weights are starting points, not tuned constants. The acceptance criterion is not lower JEPA loss; it is higher content-shuffle gap, healthier effective rank, and better downstream transfer.

### Required Diagnostics

At every saved checkpoint and at the end of every training epoch or update block, log native-SSL4EO diagnostics on fixed validation sets:

| Validation Set | Size | Purpose |
|---|---:|---|
| `generic_common_heldout` | all available common heldout samples, about `655` | continuity with `[9]` verification |
| `agriculture_heldout` | at least `2048` samples | crop-relevant content sensitivity |

A healthy candidate should satisfy all of the following:

- shuffled-target loss is materially higher than normal target loss;
- effective rank and feature variance remain healthy throughout training;
- same-time embeddings from different samples do not converge toward identical representations;
- clean/degraded displacement decreases without erasing sample-specific content;
- downstream AUROC improves or remains stable as representation diversity improves.

Near-zero JEPA loss with a near-zero shuffled-target gap is a failed representation, regardless of clean downstream F1.

Required output files:

| File | Contents |
|---|---|
| `train_history.json` | loss, learning rate, mode counts, diagnostic summaries by checkpoint |
| `checkpoint_diagnostics.csv` | content-shuffle, missingness-shuffle, effective-rank, clean-zero, variance, and covariance metrics |
| `corruption_diagnostics.csv` | full-clean and retained-timestep displacement for each corruption mode |
| `covariance_spectra.json` | context, target, and spatial residual spectra |
| `probe_results.csv` | all CropHarvest and EuroCropsML probe rows |
| `probe_summary.csv` | grouped summaries |
| `priority_summary.csv` | final compact decision table |
| `run_manifest.csv` | arm, seed, data source, fixed updates, checkpoints, and status |

### Evaluation

CropHarvest evaluation must remain identical to `[9]`:

- five strict geographic holdouts;
- clean, S2-off, S1-off, temporal-drop-50, S2-off-plus-temporal-drop, and S1-off-plus-temporal-drop conditions;
- clean-trained degraded-test and condition-matched protocols;
- default F1, calibrated F1, AUROC, raw-stat probes, and hybrid probes.

EuroCropsML evaluation should use the official cross-country few-shot split:

- pretraining or representation fitting on Latvia and Portugal only where permitted by the protocol;
- evaluation on Estonia;
- frozen linear probes and controlled few-shot fine-tuning;
- shared Sentinel-2 bands with NDVI computed exactly once;
- S1 explicitly marked unavailable;
- actual timestamps, sequence padding, and availability masks;
- temporal-drop evaluation;
- raw-feature and randomly initialized baselines.

For `[10]`, EuroCropsML should be an external evaluation benchmark, not an additional SSL pretraining corpus. If transfer is promising, long-sequence EuroCropsML pretraining or joint pretraining becomes a later experiment.

EuroCropsML is S2-only. Do not evaluate S2-off there. Evaluate:

- clean;
- temporal-drop-25;
- temporal-drop-50;
- sparse-label budgets;
- crop-type macro-F1, balanced accuracy, and top-k accuracy if multiclass class count is large.

References:

- EuroCropsML documentation: <https://eurocropsml.readthedocs.io/en/latest/>
- EuroCropsML paper: <https://arxiv.org/abs/2407.17458>
- EuroCropsML dataset: <https://zenodo.org/records/12168505>

### Expected Outcomes And Decisions

| Observation | Interpretation |
|---|---|
| `C_generic_step_matched` approaches `B_mixed_viewdrop_sensor_dropout` | mixed-pool gains are mostly optimizer exposure |
| `D_mixed_size_matched` beats `C_generic_step_matched` | agriculture mixture helps independently of sample count |
| `B_mixed_viewdrop_sensor_dropout` beats `A_mixed_viewdrop_reference` on S1-off without losing S2-off | S1-off should remain in the final robustness objective |
| `E_spatial_no_consistency` has strong content diagnostics | spatial tokens are useful once current consistency is removed |
| `F_spatial_content_regularized` beats `E` with healthier diagnostics | content-diversity regularization is a useful architectural contribution |
| CropHarvest improves but EuroCropsML transfer is weak | The learned representation remains benchmark-specific |
| EuroCropsML transfer is strong | The temporal representation captures externally useful agricultural structure |
| Spatial JEPA loss approaches zero without shuffled-target separation | The spatial objective is solving a positional shortcut and must not be scaled |

### Success Criteria

Minimum useful result:

- the `A/B/C/D` comparison resolves update exposure, sample-count, and data-composition confounds;
- the `E/F` comparison determines whether spatial JEPA can be made content-sensitive without global consistency;
- results for the reference candidates are stable across three seeds;
- native-SSL4EO content diagnostics rule out trivial position-only prediction;
- at least one learned representation beats raw and randomly initialized baselines on EuroCropsML cross-country few-shot transfer.

Strong result:

- clean CropHarvest AUROC remains at or above `0.75`;
- S2-off-plus-temporal-drop AUROC exceeds `[9] C`'s `0.684` without losing default-F1 compatibility;
- S1-off AUROC and F1 become usable rather than catastrophic;
- the selected spatial candidate preserves a substantial shuffled-target gap and healthy effective rank;
- the same candidate improves external EuroCropsML frozen or few-shot transfer.

Failure criteria:

- any arm has near-zero JEPA loss and near-zero content-shuffle gap at the final checkpoint;
- spatial same-slot cosine approaches `1.0` while clean-zero token cosine approaches `1.0`;
- S1-off training destroys S2-off robustness;
- EuroCropsML raw features beat all learned embeddings;
- the mixed-pool advantage disappears under update-matched controls.

### Decision After `[10]`

If `F_spatial_content_regularized` wins both CropHarvest and EuroCropsML with healthy diagnostics, it becomes the architecture to scale and the base for climate reintegration. If pooled `B` remains stronger externally, retain the pooled temporal model while redesigning spatial prediction before scaling. If no learned arm transfers to EuroCropsML, stop adding architecture and investigate preprocessing, sensor harmonization, and benchmark-specific shortcuts first.
