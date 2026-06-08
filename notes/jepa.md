# JEPA Notes

These notes track JEPA work that is relevant to SURF. The useful design principle is simple: predict latent targets, not pixels, so the model spends capacity on stable semantic or physical structure rather than high-entropy reconstruction details.

Web/source refresh: 2026-06-05. Primary sources are kept as direct URLs so numbers and claims can be rechecked.

## Current Implementation Reality

The current local model is a useful CropHarvest-scale prototype, not yet the higher-dimensional architecture we probably want.

Current data path:
- `src/cropharvest.py` converts CropHarvest point time series into Zarr tensors shaped `N x T x C x 1 x 1`.
- Active modalities are Sentinel-1 `VV/VH`, Sentinel-2 bands `B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12`, and two ERA5 climate variables: `temperature_2m`, `total_precipitation`.
- The CropHarvest raw arrays also contain SRTM/elevation-style channels and NDVI, but the current converter does not feed those to the model.
- The current time axis is monthly/regularized; DOY is synthetic month-center timing, not actual acquisition timing.
- Spatial context is effectively absent because patches are `1x1`.

Current model path:
- `LocalFusionEncoder`: separate small Conv2d encoders for S2, S1, and climate, followed by learnable modality gates and a weighted fusion.
- Because the current patches are `1x1`, the Conv2d patch encoder is mostly acting like a small channel MLP; it is not learning real spatial structure.
- `CausalContextEncoder`: causal Transformer over the time axis.
- Target branch: EMA copy of the local per-timestep encoder only.
- Predictor: small MLP predicting next-timestep latent target from causal context.
- Loss: cosine distance between predicted and target next-step embeddings.

Current gap: this is a causal temporal JEPA over point-level monthly features. It is not yet I-JEPA/V-JEPA-style multi-block masked latent prediction over spatial-temporal patch tokens.

## Working Research Frame

The sharp direction is:

> Apply JEPA-style latent prediction with phenological masking and strong missing-modality training to agricultural satellite time series, then show the learned representations are more robust to real agricultural failure modes: cloud gaps, sparse labels, irregular sampling, sensor dropout, and cross-region transfer.

Koopman-JEPA (AAAI 2026) theoretically validates this direction: it proves JEPA's predictive objective drives encoders to learn Koopman eigenfunctions (dynamical invariants), causing representations to cluster by dynamical regime without supervision. For agriculture, different crop types have different Koopman operators governing their phenological dynamics. This implies our embeddings should cluster by crop type / phenological regime, not by geography — test this directly with UMAP visualization of learned embeddings.

The agricultural EO input should eventually be `T x H x W x C`, not only `T x C`. A field or pixel neighborhood has spatial structure, phenological timing, cloud masks, radar/optical complementarity, climate context, and location-dependent growing conditions. The higher-gradient architecture direction is therefore to move from pooled point sequences to dense spatiotemporal tokens.

## What JEPA Sources Actually Teach

| Work | Source | Useful details for SURF |
|---|---|---|
| I-JEPA | https://arxiv.org/abs/2301.08243 | Predicts target-block representations from a context block, not pixels. The source emphasizes that target blocks must be large enough to be semantic and the context must be spatially distributed/informative. Uses a context encoder, EMA target encoder, narrow ViT predictor, mask/position tokens, and no hand-crafted view augmentations. It reports ViT-H/14 ImageNet training on 16 A100s in under 72 hours. |
| V-JEPA | https://arxiv.org/abs/2404.08471 | Strongest base template for our next architecture. Trains only with feature prediction: no pretrained image encoder, text, negatives, labels, or pixel reconstruction. Uses spatiotemporal ViT tokens, drops masked tokens from the encoder, concatenates learned mask tokens into a narrow predictor, and regresses target encoder features. Multi-block masking wins over random-tube and causal-only masks. |
| V-JEPA 2 | https://arxiv.org/abs/2506.09985 | Scaling lesson: V-JEPA 2 pretrains on over 1M hours of internet video/image data and scales encoders from ViT-L 300M to ViT-g 1B while keeping the predictor fixed at ViT-s 22M. It uses 16-frame clips first, then a cooldown phase with 64 frames and higher crop sizes. Key architecture details include tubelet size 2, patch size 16, high EMA, 3D positional handling, and block-causal attention in action-conditioned post-training. |
| VJ-VCR | https://arxiv.org/abs/2412.10925 | Video JEPA with variance-covariance regularization to avoid collapse and latent variables for uncertain futures. Useful if deterministic next-latent regression underfits stochastic crop/weather futures. |
| DMT-JEPA | https://arxiv.org/abs/2405.17995 | Calls out a weakness of vanilla JEPA: latent masked modeling can lose local discriminative semantics. DMT-JEPA builds targets from semantically similar neighboring patches using feature similarity and lightweight cross-attention. This maps cleanly to our LEM Brazil issue: preserve local red-edge/NIR crop-discriminative cues instead of smoothing them away. |
| MC-JEPA | https://arxiv.org/abs/2307.12698 | Jointly learns content and motion/flow features. For crops, the analogue is content plus phenological change: predict not only state embeddings but growth/change embeddings, vegetation-index derivatives, or stress-transition latents. |
| LeJEPA / SIGReg | https://arxiv.org/abs/2511.08544 | Proposes a regularized JEPA objective based on an isotropic Gaussian embedding target, claiming simpler and more stable training without stop-gradient/teacher-student heuristics. High-value as a controlled anti-collapse ablation, but not the next default replacement for EMA until `[10]` tells us whether the EMA target path is actually the limiting factor. |
| EB-JEPA library | https://arxiv.org/abs/2602.03604 | Practical reference library for energy-based JEPA components, video examples, and action-conditioned world-model examples. Useful as implementation scaffolding or ablation inspiration, not as the main method target. |
| VJEPA probabilistic | https://arxiv.org/abs/2601.14354 | Variational/probabilistic JEPA. Relevant later for uncertainty-aware crop stress or yield-risk predictions, where one context can imply multiple plausible futures. |
| TS-JEPA | https://arxiv.org/abs/2406.04853 | Predicts future low-dimensional semantic embeddings from current embeddings and commands in capacity-limited control settings. Not EO, but supports the idea that future-embedding prediction can be useful when raw reconstruction is unnecessary. |
| MTS-JEPA | https://arxiv.org/abs/2602.04643 | Multi-resolution time-series JEPA with a soft codebook bottleneck to separate transient shocks from long-term trends. This is very aligned with phenology plus stress/anomaly modeling. |
| NEPA | https://arxiv.org/abs/2512.16922 | Next-Embedding Predictive Autoregression: simplifies JEPA to a single shared embedding layer + autoregressive transformer predictor with causal masking and stop-gradient — no asymmetric branches, no separate prediction head, no pixel reconstruction, no contrastive loss. Predicts future patch embeddings from past ones. Achieves 83.8% / 85.3% top-1 on ImageNet-1K with ViT-B/L after fine-tuning. Relevant as a simpler, scalable alternative if JEPA's asymmetric design proves unnecessary for our setup. |
| Koopman-JEPA | https://arxiv.org/abs/2511.09783 (AAAI 2026) | Proves JEPA's predictive objective drives it to learn the invariant subspace of a system's Koopman operator for time-series data. The encoder learns regime indicator functions that are Koopman eigenfunctions (eigenvalue 1). For agriculture: different crops have different Koopman operators governing phenological dynamics → JEPA representations should cluster by crop type without supervision. Validates our JEPA choice theoretically; use as motivation for UMAP analysis showing clustering by phenological regime, not geography. |
| VL-JEPA | https://arxiv.org/abs/2512.10942 (ICLR 2026) | Predicts continuous text embeddings from image/video context instead of autoregressive token generation. Uses x-encoder (vision), y-encoder (text), and predictor mapping visual → text embeddings in latent space. Achieves stronger performance than standard VLMs with 50% fewer parameters, with selective decoding (~2.85x fewer ops). Agricultural analogue: predict crop type name embedding from satellite time series. If combined with channel description conditioning (SLIP-style), becomes directly relevant for predicting what spectral band descriptions would say about a field's state. |
| ACT-JEPA | https://arxiv.org/abs/2501.14622 (2025) | Combines JEPA world model with imitation learning for robot policy. Jointly predicts action sequences and latent observation sequences, filtering irrelevant details. Reports 40% improvement in world model understanding and 10% higher task success over IL baselines. Relevant for future decision-support framing: if our model predicts crop-state trajectories (the world model), an action predictor could recommend irrigation timing or harvest dates. |
| Var-JEPA / Var-T-JEPA | https://arxiv.org/abs/2603.20111 (ICML 2026) | Reframes JEPA as a variational latent-variable model with a unified ELBO objective. Var-T-JEPA instantiation handles heterogeneous tabular data with tokenization, Transformers, and principled KL regularization instead of ad-hoc anti-collapse heuristics (EMA, variance penalties). Consistently improves over T-JEPA; discarding the most uncertain 10-50% of samples improves accuracy by 4-8%. Our CropHarvest data is tabular-format (each row is a pixel time series with mixed-modality channels). Check their feature-level masking strategy and heterogeneous channel handling — likely directly applicable to our S1/S2/climate mixed-channel input. |
| EC-IJEPA (Apple) | https://arxiv.org/abs/2410.10773 (2024) | Conditions the context encoder with target-window positions and the target encoder with context-window positions to prevent representational collapse without extra loss terms. Based on the intuition that spatially local regions have higher mutual information. Shows improved robustness to context window size, better sample efficiency, and gains on ImageNet classification. Directly relevant as a simpler collapse-prevention mechanism for our spatiotemporal setup. |
| DSeq-JEPA | https://arxiv.org/abs/2511.17354 (2025) | Discriminatively ordered sequential prediction: uses attention-derived saliency maps to identify primary discriminative regions, then predicts subsequent regions sequentially from most to least discriminative (easy-to-hard curriculum). Consistently outperforms I-JEPA on ImageNet, fine-grained categorization, detection/segmentation, and CLEVR reasoning. Maps to our phenological masking: mask high-information phenological transition periods first, creating a natural curriculum from discriminative growth-stage transitions to background/fallow periods. |
| X-JEPA | https://openaccess.thecvf.com/content/WACV2026/papers/Choudhury_X-JEPA_A_Novel_Joint_Learning_Cross-Modal_Predictive_Alignment_Framework_for_WACV_2026_paper.pdf | Cross-modal remote-sensing JEPA for retrieval. The key idea is predicting the semantic embedding of one modality from another, plus prediction-space alignment, instead of reconstructing pixels. This is directly relevant to cloud-gap robustness: S1/context should predict an S2 latent target when optical observations are missing. |
| M3-JEPA | https://proceedings.mlr.press/v267/lei25b.html | ICML 2025 multimodal JEPA that uses a multi-gate mixture-of-experts predictor for modality alignment. Relevant if the shared predictor in SURF cannot handle all missing-modality regimes equally well. The risk is routing collapse or extra instability; treat as a second-wave predictor ablation, not the first fix. |
| SALT | https://openreview.net/pdf?id=3cB9243E9i | Static-teacher asymmetric latent training. It replaces the moving EMA teacher with a frozen teacher trained by pixel reconstruction, then trains a student to predict the teacher's latents under V-JEPA-style masking. Useful if EMA targets stay unstable or too smooth. It is heavier than SIGReg/VICReg because it requires a separate teacher stage. |
| Polymer-JEPA | https://arxiv.org/abs/2506.18194 (2025/2026) | Applies JEPA with GNN encoders (weighted directed MPNN) for self-supervised pretraining on polymer molecular graphs. Predicts embeddings of target subgraphs from context subgraphs in latent space, avoiding reconstruction of noisy molecular details. Pretraining improves downstream property prediction by up to 39.8% in label-scarce scenarios (~430 labels). Molecular graphs share structural similarity with crop phenological networks: both have regular phase transitions governed by underlying physics. |

Highest-value JEPA additions for SURF, ranked:

1. Spatiotemporal multi-block masking. Move from next-step-only prediction to V-JEPA/I-JEPA-style prediction of multiple masked blocks across time, space, and modality. Use target masks such as whole phenological windows, clouds over contiguous time spans, and spatial crop/non-crop boundary blocks.
2. Saliency-ordered curriculum masking. DSeq-JEPA's discriminatively ordered sequential prediction (masking high-information regions first) maps directly to phenological transition periods. Mask growth-stage transitions before background/fallow periods — creates an easy-to-hard curriculum from discriminative phenophase transitions to stable periods.
3. Mask-conditioned transformer predictor. Replace the current MLP predictor with a small transformer predictor that receives context tokens plus learned mask tokens containing time, space, modality, and band/group position.
4. Full spatiotemporal EMA target encoder. The current target encoder is local per timestep. For patch data, target features should come from an EMA encoder that has seen the full unmasked spatiotemporal crop/patch sequence.
5. Multi-horizon latent prediction. Predict next month, next phenological phase, and end-of-season summary targets. V-JEPA 2 action anticipation results scale with future prediction, but longer horizons get harder; agriculture gives us natural horizon structure.
6. Cross-modal latent prediction. Train S2-from-S1/context, S1-from-S2/context, and climate/phenology-from-sensors targets. Do this in latent space, not full-pixel reconstruction. X-JEPA is the closest direct precedent.
7. EC-IJEPA-style positional conditioning. Condition encoders on context/target positions to make the model aware of where prediction is happening without relying on predictor-only position shortcuts.
8. Local discriminative target construction. Use DMT-JEPA-style neighbor targets or auxiliary raw-stat targets to preserve red-edge/NIR/NDVI cues. This is directly motivated by LEM Brazil, where simple raw NIR/red-edge features beat all learned embeddings.
9. Anti-collapse regularization alternatives. Keep variance/covariance as the conservative baseline. Test SIGReg only as a small controlled ablation after `[10]`, not as a wholesale EMA replacement before we know EMA is the bottleneck.
10. Multi-resolution temporal objective. Split representations into short-term shock tokens and seasonal trend tokens. MTS-JEPA's shock/trend framing is a good conceptual fit for drought, cloud gaps, senescence, planting, and harvest.
11. Modality-routed predictor. Use M3-JEPA-style routing if shared predictor performance diverges strongly by condition, especially S2-off versus S1-off.
12. Static-teacher target path. SALT is worth testing only if moving EMA targets are noisy/unstable or too smooth after `[10]` diagnostics; otherwise it adds an extra teacher-stage cost.
13. Heterogeneous tabular/feature-level masking. Var-T-JEPA's principled KL-regularized approach for mixed-modality tabular data is close to CropHarvest point data. Use it if we need a point-sequence fallback or climate-feature masking.
14. Dense feature probing. V-JEPA-style dense patch features matter; evaluate both pooled field embeddings and per-pixel/per-patch downstream heads.
15. Interpretability decoder. Train a lightweight frozen-encoder decoder or linear raw-stat probes to inspect whether embeddings preserve NIR, red-edge, NDVI, S1 backscatter, cloud/missingness, and phenology.
16. ACT-JEPA-style action prediction (future). Once the world model is solid, add an action predictor head for decision-support tasks like irrigation timing and harvest date recommendation.
17. VL-JEPA-style open-vocabulary conditioning (future extension). Predict crop-type name embeddings or channel-description embeddings from time-series context — enables open-vocabulary classification without fixed crop-type heads.

## Current JEPA Decision Notes

These are current operational opinions, not literature summaries.

### SIGReg / LeJEPA

SIGReg is worth taking seriously because it attacks the weakest piece of vanilla JEPA: EMA plus stop-gradient is a working heuristic, not a clean principle. The LeJEPA result claims a simpler objective with no teacher-student branch and an explicit isotropic-Gaussian embedding constraint.

For SURF, it should not replace EMA in the next main run by default. `[10]` is already testing whether the spatial objective and content regularization repair the failures from `[8]` and `[9]`. Replacing EMA at the same time would confound the result. The right first SIGReg experiment is a narrow `[11]`-style ablation:

| Arm | Change | Decision signal |
|---|---|---|
| EMA baseline | best `[10]` architecture unchanged | reference |
| EMA + SIGReg | add SIGReg to sequence/global embeddings | if collapse diagnostics improve without hurting transfer |
| no-EMA SIGReg | remove EMA target only if the previous arm works | if teacher branch is unnecessary |

SIGReg should be judged by downstream transfer, effective rank, content-shuffle gap, and raw-cue retention. Lower JEPA loss alone is not evidence.

### X-JEPA

X-JEPA is the strongest near-term architectural idea after `[10]` because it directly trains the deployment failure mode: optical missingness. The SSL objective should include a cross-modal latent term such as:

```text
context: S1 + timing + optional climate/location
target:  S2 latent from full target encoder
loss:    predict target S2 latent under cloud/S2-off context
```

This is more direct than generic modality dropout. Modality dropout asks the shared predictor to survive missing inputs; cross-modal JEPA asks it to infer the missing modality's state in latent space.

### DMT / Raw-Cue Targets

DMT-JEPA matters because the LEM Brazil failure is not only a classifier or threshold issue. Raw red-edge/NIR/SWIR cues retain information that the learned embedding discards. A DMT-style target or raw-stat auxiliary target should force the target representation to keep local discriminative agronomic structure.

The first version should be simple:

- predict a compact raw-cue vector alongside the latent target;
- raw cues: `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`, `NDVI`, `VV`, `VH`;
- compare embedding-only, raw-only, and hybrid probes on LEM Brazil and EuroCropsML.

Neighbor-aggregated targets can come later if the auxiliary target works.

### Climate Reintegration

Climate should come back after the `[10]` data/evaluation path is stable. The right order is:

1. Use climate in downstream/eval datasets only as masks and optional raw probes.
2. Add climate as a context modality in SSL4EO pretraining only when there is a real temporal join, not synthetic repeated values.
3. Test climate in three modes: context-only, target-predicted, and held-out/dropout.

Climate is important for the final claim, but a sloppy join would create another confound. It should be a deliberate experiment, not a quick channel append.

## EO Architecture Lessons

| Work | Source | Useful details for SURF |
|---|---|---|
| AnySat | https://arxiv.org/abs/2412.14123 | Direct JEPA-based EO neighbor. It uses JEPA plus scale-adaptive spatial encoders for heterogeneous resolutions, scales, and modalities. It trains one model on GeoPlex, a 5-dataset, 11-sensor collection, and reports external tasks including crop type classification and flood/burn/deforestation segmentation. Our distinction should be temporal/phenological crop dynamics and climate-aware missing-sensor robustness. |
| CROMA | https://arxiv.org/abs/2311.00566 | Radar-optical lesson: separate masked optical and SAR encoders, cross-modal contrastive learning, then fused multimodal encodings for masked-patch prediction. It also adds spatial attention biases that allow extrapolation to much larger test images. Good blueprint for S1/S2 fusion without forcing one modality to become the other. |
| Galileo | https://arxiv.org/abs/2502.09356 | Generalist multimodal pressure. It explicitly handles optical, SAR, elevation, weather, pseudo-labels, and more across space and time. Its global and local losses differ in targets and masking strategies. This argues for a local/dense head and a global/field head in SURF rather than a single pooled embedding. |
| OlmoEarth | https://arxiv.org/abs/2511.13655 | Our strongest current clean-transfer pressure. It is designed for EO's spatial, sequential, and multimodal nature, and reports best embedding performance on 15/24 tasks plus best fine-tuning on 19/29. We should treat it as a target geometry to learn from, but [5] showed its stress behavior still needs careful validation in our code path. |
| SSL4EO-S12 v1.1 | https://arxiv.org/abs/2503.00168 | Best immediate data-scaling target: 246k time series and nearly 1M image patches, stored as WebDataset/Zarr, with cloud masks and added elevation, land-cover, and vegetation modalities. This is the natural next unlabeled corpus if disk and download are manageable. |
| Prithvi-EO-2.0 | https://arxiv.org/abs/2412.02732 | Scaling reference: 4.2M global HLS time-series samples, 300M/600M models, temporal and location embeddings. It does not match our S1/S2/climate mix exactly, but it proves temporal/location embeddings and large optical patch sequences are now table stakes. |
| Presto | https://arxiv.org/abs/2304.14065 | Direct pixel-time-series baseline. It is lightweight, uses remote-sensing-specific inputs and missing-data handling, and is operationally relevant. [5] showed Presto has strong AUROC but weak F1 under our strict heldout setup, so threshold/calibration and downstream probing protocol matter. |
| WorldCereal deployment lessons | https://openreview.net/forum?id=eHW9HWitP0 | Real-world reminder: standardized benchmark wins are not enough. Deployment needs domain-specific adaptation, empirical testing under data heterogeneity, resource constraints, and application requirements. This should guide our robustness and sparse-label setup. |
| Mamba-2 / SSD | https://arxiv.org/abs/2405.21060 | Not JEPA-specific, but relevant if long Sentinel sequences make quadratic attention painful. The source connects Transformers and SSMs through structured state-space duality and reports Mamba-2 as 2-8x faster than earlier Mamba. Treat as an encoder option after the tokenization/objective is settled. |

## Landscape Threads Around The EO Architecture Papers

The useful author/linkage pass is not an exhaustive author bibliography. The value is to identify research clusters that already occupy pieces of the claim space.

| Thread | Related sources | What is already occupied | Implication for SURF |
|---|---|---|---|
| Crop sparse-label / deployment line | TIML: https://arxiv.org/abs/2202.02124; Rapid-response Togo crop maps: https://arxiv.org/abs/2006.16866; Presto: https://arxiv.org/abs/2304.14065; WorldCereal deployment: https://arxiv.org/abs/2508.00858; Cropland embeddings in Togo: https://arxiv.org/abs/2511.02923 | This line already owns the practical story around smallholder crop mapping, sparse labels, transfer across data-rich/data-poor regions, Presto-style pixel time series, and operational WorldCereal-style deployment constraints. | We should not frame SURF as merely "better crop mapping." The wedge has to be representation robustness: cross-region transfer, strict sparse-label curves, missing sensors, irregular timing, and crop-specific temporal stress behavior. |
| Presto / Galileo / OlmoEarth generalist line | Presto: https://arxiv.org/abs/2304.14065; Galileo: https://arxiv.org/abs/2502.09356; OlmoEarth: https://arxiv.org/abs/2511.13655; LFMC maps: https://arxiv.org/abs/2506.20132 | This line already shows that pixel-time-series, multimodal local/global features, and pretrained geospatial embeddings can be useful across agriculture and vegetation-monitoring use cases. | The direct comparison needs to include Presto-style pixel probes and OlmoEarth/Galileo-style shared transfer tables. Our novelty must come from the objective/input design and the failure-mode evaluation, not from having multiple modalities. |
| AnySat / OmniSat heterogeneous-modality line | OmniSat: https://arxiv.org/abs/2404.08351; AnySat: https://arxiv.org/abs/2412.14123 | This line already covers self-supervised modality fusion, heterogeneous EO sensors, scale-adaptive encoders, and JEPA for many resolutions/scales/modalities. | Do not claim "JEPA for multimodal EO" as the core open space. AnySat is too close. The open space is crop-temporal JEPA with phenology-aware masks, climate context, and explicit robustness curves. |
| CROMA radar-optical fusion line | CROMA: https://arxiv.org/abs/2311.00566; SSL4EO-S12: https://arxiv.org/abs/2211.07044; SSL4EO-S12 v1.1: https://arxiv.org/abs/2503.00168 | This line already shows radar-optical contrastive fusion plus masked-patch prediction can learn strong Sentinel-1/2 representations at SSL4EO scale. | The first bigger data run should borrow the S1/S2 fusion lesson, but add crop timing, climate/terrain variables, and missingness training. A plain S1/S2 MIM model is not enough of a wedge. |
| Broad multimodal / benchmark line | TerraMind: https://arxiv.org/abs/2504.11171; Copernicus-FM: https://arxiv.org/abs/2503.11849; TerraFM: https://arxiv.org/abs/2506.06281; THOR: https://arxiv.org/abs/2601.16011; PANGAEA: https://arxiv.org/abs/2412.04204; No One Knows: https://arxiv.org/abs/2605.12678 | This line already owns very large multimodal pretraining, all-Sentinel handling, flexible resolution, missing-modality/generative objectives, and broad GFM evaluation criticism. | Our evaluation needs copied-vs-rerun annotations, shared splits, variance, and data-vs-architecture-vs-objective controls. The narrow claim space is agricultural reliability under realistic data failure modes. |
| Hyperspectral / spectral richness line | SpectralEarth: https://arxiv.org/abs/2408.08447; SpectralGPT: https://arxiv.org/abs/2311.07113; cereal HSI benchmark: https://arxiv.org/abs/2510.11576 | This line warns that crop-type and stress tasks may be dominated by spectral richness when hyperspectral inputs exist. | Sentinel-only SURF must preserve red-edge/NIR/SWIR and radar cues very deliberately. If stress/yield proxies become the main downstream task, we should consider hyperspectral or at least stronger raw spectral auxiliary targets. |
| Long-sequence efficiency line | Mamba-2 / SSD: https://arxiv.org/abs/2405.21060; MTS-JEPA: https://arxiv.org/abs/2602.04643 | Efficient sequence models and multi-resolution time-series objectives are plausible upgrades once sequences get longer. | Useful after patch-token data is working. It is premature to spend the main run on SSM architecture before fixing input dimensionality and objectives. |

## Input Expansion Plan

The next architecture should think in tokens, not rows.

Recommended input structure:
- Spatial tokens: `H x W` patch tokens, at least `16x16` or `32x32` for pilot; larger later if memory allows.
- Temporal tokens: actual acquisition dates when available, not only monthly bins; include DOY, month, and optionally growing-degree-day or agro-climatic phase.
- Spectral tokens/groups: preserve visible, red-edge, NIR, SWIR, radar, climate, elevation/terrain, vegetation indices, cloud/quality masks.
- Modality tokens: S2, S1, climate, terrain, vegetation/land-cover priors, and missingness indicators should each have explicit identity/availability embeddings.
- Location tokens: latitude/longitude or coarse agro-ecological region embeddings; Prithvi-EO-2.0 and AlphaEarth-style mapping make location context hard to ignore.
- Raw-stat side channel: mean/min/max or quantiles for B5/B6/B7/B8/B8A/B11/B12, NDVI, and VV/VH, either as auxiliary prediction targets or appended to downstream probes.

Architecture sketch:
- Per-modality patch tokenizer for S2, S1, climate/terrain, and optional priors.
- Shared spatiotemporal Transformer over visible context tokens.
- Small mask-conditioned predictor Transformer with learned time/space/modality mask tokens.
- EMA target encoder over the unmasked full sample.
- Two heads for downstream use: dense local tokens and pooled field/sequence embedding.

## Highest-Gradient Direction After The Current Point-Time-Series Runs

The next major run should not simply scale width/depth on the current `1x1` CropHarvest runner. Scaling that architecture mostly scales a bottleneck.

Recommended [7] direction:

1. Build a real patch-time-series JEPA runner on SSL4EO-S12 v1.1 plus any local CropHarvest-adjacent chips we can assemble.
2. Use V-JEPA/I-JEPA-style multi-block masking over `time x space x modality`, with a small transformer predictor.
3. If raw-cue preservation returns, treat it as a fresh clean experiment, because LEM Brazil says learned embeddings are currently discarding useful red-edge/NIR separability.
4. Add a local/global split: dense token loss plus pooled field/sequence loss.
5. Evaluate two frozen outputs: pooled embedding and dense embedding with a light segmentation/classification head.

The scaling ladder should be:
- `Point-JEPA`: current CropHarvest setup, mainly diagnostic.
- `Patch-JEPA-S`: `16x16`, 12-24 timesteps, S1/S2/cloud/masks, 100M-ish model.
- `Patch-JEPA-B`: `32x32`, richer modalities, 200-300M-ish model.
- `Agro-JEPA-L`: add climate/terrain/location/phenology, multi-resolution objectives, and longer horizon prediction.

## Do Not Overfit To JEPA Branding

The goal is not to make the most ornate JEPA. The highest-gradient bets are:
- richer input dimensionality,
- structured spatiotemporal masking,
- target representations that preserve agricultural raw cues,
- explicit cross-modal and missing-modality training,
- robust evaluation under heldout geography/dataset, sparse labels, and sensor/time dropout.

If a non-JEPA component improves those properties, use it.
