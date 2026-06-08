# Agent Notes

This is a small working-memory file for future maintenance passes. It is intentionally ignored by git via `notes/` in `.gitignore`.
## Current Mission
The mission is to build a JEPA-based agricultural Earth observation model that is strongest on crop-temporal representation learning under realistic agricultural deployment failures: strict geographic/dataset transfer, sparse labels, missing sensors, and irregular temporal coverage.

Do not frame the project as "beat every EO foundation model everywhere." Broad multimodal EO is already crowded. The target is narrower and sharper: show that a lightweight crop-temporal model can outperform direct pixel-time-series baselines and compete meaningfully with much larger general EO models on the agricultural failure modes that matter.

The key notes files are:
- `notes/goal.md`: long-term project compass, input/output definition, differentiation, roadmap, and limits.
- `notes/competitors.md`: scoreboard for competing models and reported results.
- `notes/jepa.md`: JEPA method notes and design implications.
- `notes/benchmarks.md`: benchmark/data access notes.
- `notes/goal.md`: long-term project compass, including the explicit ICLR-shaped ambition, evidence bar, and paper framing.
- `notes/agents.md`: this file.
## Local Repo State To Remember
- README has been rewritten around the JEPA/agricultural EO benchmark mission.
- Old planning docs were folded into `notes/` or removed.
- `source_links_2026-04-10.csv` has been deleted after preserving useful links in notes.
- `notes/` is gitignored, by user request.
- Existing experiment artifacts under `artifacts/figures/` and large data under `data/` are ignored by `.gitignore`.
## Available Training Machine Mentioned By User

Remote host `digital-ag`:
- CPU: AMD Ryzen 9 5950X, 16 cores / 32 threads.
- RAM: 62 GiB.
- GPU: 2x NVIDIA RTX 3090, 24 GiB each.
- Disk: `/home` was tight at 89% used; `/mnt/data_drive_temp` has a large 14.6T disk.
- No cluster scheduler detected.

Practical implications:
- Use explicit GPU selection for two-GPU runs.
- Prefer checkpoints/logs/data on the large mounted data drive instead of `/home`.
- Be careful with many small files and dataloader workers; open-file limit was 1024.
- For long jobs, use `tmux`, `nohup`, or a small process supervisor.

## Performance Notes For Future Runs

### `digital-ag` Runtime Defaults

The useful machine budget is `2` GPU training workers plus about `32` CPU threads total. Do not assume that higher GPU count automatically means higher throughput; the current SSL4EO-style stores can become CPU/I/O bound before the GPUs are saturated.

Good starting point for two-GPU PyTorch runs on this host:

```python
TRAIN_NUM_WORKERS = 8
EVAL_NUM_WORKERS = 4
DIAGNOSTIC_NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 4
```

This creates roughly `16` active training DataLoader workers across the two GPU processes, and roughly `8` workers during evaluation. Evaluation creates many short-lived loaders and also runs sklearn/NumPy-heavy probe code, so it must be more conservative than training.

Avoid blindly increasing DataLoader workers after this point. If CPU load is already around `16-32` and GPU utilization is still low, the bottleneck is probably decompression, small reads, or collation overhead. More workers can make that worse.

### Interpreting Low GPU Utilization

Low GPU utilization during training is a throughput problem, not automatically a correctness problem.

Check these before interrupting:

```bash
ssh digital-ag 'cd /home/agarapat/surf && pid=$(cat logs/run_10.pid) && ps -p "$pid" -o pid=,etime=,stat=,%cpu=,%mem=,args= && tail -n 80 logs/run_10.log && nvidia-smi'
```

If the log has reached the previous scheduled checkpoint and no later checkpoint is due yet, silence can be normal. For `[10]`, the schedule is:

```python
CHECKPOINT_STEPS = [1_000, 3_000, 10_000, 16_600]
```

So there is expected silence between `3_000` and `10_000`.

### File Descriptor Limit

The soft open-file limit on `digital-ag` was `1024`. This is a risk for many-small-file formats, but it is not automatically the bottleneck.

Launch checklist for larger PyTorch runs:

- Set `torch.multiprocessing.set_sharing_strategy("file_system")` before DataLoader workers are created.
- Raise the soft open-file limit to at least `8192` before launch, or do it inside the runner with `resource.setrlimit`.
- Confirm the launched process inherited the higher limit from `/proc/<pid>/limits`.
- Cap CPU thread pools before importing NumPy/sklearn/torch: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`.
- Use `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` inside large multiprocessing runners.
- Do not use persistent DataLoader workers in evaluation helpers that are repeatedly constructed inside loops. Persistent eval loaders can leak hundreds of pipe fds per worker process.
- If a run crashes with `Too many open files`, preserve completed training/checkpoint outputs and relaunch eval only; do not delete complete training dirs.
- If load average explodes while GPUs are at `0%` and output files/logs stop changing, check fd types. Thousands of `pipe:` fds means the eval loader/process lifecycle is broken; stop and relaunch after fixing it.

Verify actual fd usage before blaming it:

```bash
ssh digital-ag 'ulimit -n && for p in $(ps -u agarapat -o pid=,args= | awk '\''/python -u runners\/\[[0-9]+\]\.py|spawn_main/ {print $1}'\''); do c=$(ls /proc/$p/fd 2>/dev/null | wc -l); printf "%s %s\n" "$p" "$c"; done | sort -k2 -nr | head -40'
```

If processes are using tens of fds, the `1024` limit is not the active problem. If any process approaches hundreds-to-`1024`, relaunch with a higher soft limit:

```bash
ulimit -n 8192
```

### Sequential Phase Design Wastes GPU Time

Runner `[10]` uses sequential phases: train all arms on both GPUs first, then eval. If one GPU finishes its arms significantly before the other (e.g. spatial arms take longer, or arm distribution is uneven), that GPU sits idle — its child becomes a zombie and the main process sleeps on `poll_schedule_timeout` waiting for the slower GPU to finish.

There is no inter-GPU synchronization or early-eval-start mechanism within a phase. All training must complete before **any** evaluation begins.

This can be fixed in future runners by:
- Starting evals per-arm as soon as training completes (not waiting for all arms).
- Distributing arms more evenly by expected runtime, not just by count.
- Using a shared work queue per GPU so a faster GPU can pick up remaining arms from the slower GPU's queue.

before starting the background job.

### Disk Placement

`/home` has been tight and has held most repo data. Root NVMe had much more space and is writable through `/tmp` and `/var/tmp`.

Observed layout:

```text
/home    -> separate NVMe, often nearly full
/        -> root NVMe, much more free
/tmp     -> writable on root NVMe
/var/tmp -> writable on root NVMe
```

For short-lived hot training stores, prefer:

```text
/var/tmp/agarapat/surf-data
```

over `/home/agarapat/surf/data` when the job is clearly data-feed bound. `/var/tmp` is safer than `/tmp` for multi-day work, but still treat it as scratch and keep canonical data under `/home` or the large mounted drive.

Do not move data paths mid-run. Stop cleanly, copy/symlink the hot stores, then relaunch from scratch or from a scientifically valid complete checkpoint boundary.

### Storage Format

Zarr is convenient and resumable for preprocessing, but it can be poor for high-throughput training when each batch triggers many small compressed chunk reads.

If the GPUs stay underfed after DataLoader worker/pinning fixes, do not keep adding workers. The next optimization should be a training-oriented cache format:

- contiguous `.pt`/`.npy` shards,
- WebDataset-style tar shards,
- or another large-shard format with fewer random opens and larger sequential reads.

The goal is fewer file opens, fewer small chunks, less Python object overhead, and larger sequential reads per batch. Keep Zarr as the canonical preprocessing artifact only if it remains fast enough.

### Evaluation And Diagnostics

Training and evaluation should be separate phases for expensive experiments:

1. Train all arms and seeds first.
2. Save scheduled checkpoints.
3. Run diagnostics and downstream probes only after training is complete.

Do not regenerate heavy diagnostics at every restart before training resumes. If partial training resume is not exact, fail loudly and delete partial run directories before relaunch.

For CPU-heavy eval phases:

- use conservative `EVAL_NUM_WORKERS` and `DIAGNOSTIC_NUM_WORKERS`; evaluation often runs more Python/sklearn CPU work than training,
- keep `PIN_MEMORY = True` when extracting embeddings on CUDA,
- persist eval metadata/schema so stale probe outputs cannot be reused silently,
- write chunk-level eval outputs as soon as each holdout/benchmark finishes, then aggregate them into the per-checkpoint CSV. Do not wait for all holdouts plus EuroCrops to finish before writing anything for a final checkpoint.
- write per-checkpoint outputs as soon as all required chunks for that checkpoint exist so a failed eval phase can be inspected without guessing.
- do not use full-batch multinomial `LogisticRegression(lbfgs, max_iter=20000)` for very large multiclass datasets. EuroCropsML has roughly `706k` rows and `176` classes; use a bounded scalable probe such as `SGDClassifier(loss="log_loss")` and log probe progress by condition/model/budget.
- sanitize multiclass `predict_proba` output before probability-based metrics. SGD/log-loss probes can occasionally return non-finite or zero-sum probability rows on large multiclass runs; repair to a valid distribution and record the repair count.

### Checkpoint And Artifact Hygiene

Checkpoints are useful while an experiment is active because they allow intermediate diagnostics, failure inspection, and final checkpoint selection. After results are summarized, old checkpoints are usually the first thing to delete.

Safe cleanup pattern:

```bash
find artifacts -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -delete
```

Keep CSV/JSON/log summaries unless there is a specific reason to delete them.

Before launching a non-resumable run, delete stale partial directories:

```bash
rm -rf "artifacts/[10]"/*_seed*
```

Do this only after confirming no current run is using those directories.

### Monitoring Commands

Process/log/GPU/disk snapshot:

```bash
ssh digital-ag 'cd /home/agarapat/surf && pid=$(cat logs/run_10.pid) && if kill -0 "$pid" 2>/dev/null; then echo "RUNNING: PID $pid"; ps -p "$pid" -o pid=,etime=,stat=,%cpu=,%mem=,args=; else echo "NOT RUNNING"; fi && echo ---LOG--- && tail -n 80 logs/run_10.log && echo ---DISK--- && df -h /home / /var/tmp && echo ---GPU--- && nvidia-smi'
```

CPU worker snapshot:

```bash
ssh digital-ag 'ps -u agarapat -o pid=,ppid=,etime=,stat=,%cpu=,%mem=,args= | egrep "runners/\\[[0-9]+\\]|spawn_main|resource_tracker" | head -100'
```

File descriptor snapshot:

```bash
ssh digital-ag 'for p in $(ps -u agarapat -o pid=,args= | awk '\''/python -u runners\/\[[0-9]+\]\.py|spawn_main/ {print $1}'\''); do c=$(ls /proc/$p/fd 2>/dev/null | wc -l); printf "%s %s\n" "$p" "$c"; done | sort -k2 -nr | head -40'
```

## Implementation Warnings
- The current `NextEmbeddingSSLModel` uses a bidirectional Transformer encoder. If used for next-step prediction, it can leak future information unless we add causal masking or separate context/target encoders.
- Self-supervised validation loss alone is not enough. Benchmark with downstream probes and public splits.
- CropHarvest conversion creates 1x1 patch tensors. Useful for quick probes, not enough for spatial foundation-model evaluation.
- The first serious method change should be a causal/target-encoder JEPA setup, not more small probe tables.
- For the next high-signal experiment, a clean ViT-S/B scale run with strong ablations is more useful than an undertrained giant model.
- For long experiments, run in the background and provide inspection commands instead of waiting interactively.

## Research Direction Notes
- Core problem: agricultural labels are sparse, expensive, and geographically biased, while unlabeled satellite observations are abundant.
- Core failure modes: cross-region domain shift, irregular sampling, cloud-blocked optical data, missing sensors, sparse downstream labels.
- Core answer: learn predictive latent representations over multi-sensor crop time series.
- Primary evidence: better few-label downstream performance and robustness curves compared with Presto, Galileo, Prithvi-style baselines, SSL4EO-S12 baselines, and task-specific temporal models.

## Paper Positioning Snapshot

Current standing after experiments `[1]-[5]`:

| Comparison | Current read |
|---|---|
| Raw flattened features | SURF is clearly better overall under strict heldout transfer: roughly `0.494` F1 vs raw `0.311`. |
| Presto | SURF beats Presto by strict-heldout F1: roughly `0.494` vs `0.390`, but Presto still has stronger AUROC: roughly `0.729` vs `0.670`. Treat threshold calibration as an active issue. |
| OlmoEarth | OlmoEarth is still ahead by clean F1 and priority F1 after the fixed stress rerun: priority F1 `0.5553` / AUROC `0.6818`; clean F1 `0.5996` / AUROC `0.7296`. |
| AnySat | AnySat already occupies the generic "JEPA for multimodal EO" lane. SURF must distinguish itself by crop-temporal dynamics, phenology-aware timing, and agricultural missingness. |
| Galileo / TerraMind / Prithvi / Copernicus-FM | These are broad EO foundation models. Use them as pressure and context; do not claim overall superiority unless we have direct matched evidence. |

Defensible current claim:

> SURF is a crop-focused temporal JEPA framework showing that lightweight multimodal representations can outperform direct pixel-time-series baselines under strict agricultural transfer and realistic sensor/time degradation, while exposing a key failure mode of learned EO embeddings: loss of simple agronomic spectral cues.

What not to claim:
- Not "best EO foundation model."
- Not "first JEPA for EO."
- Not "first multimodal S1/S2/weather model."
- Not "beats state of the art broadly."

What can be claimed now:
- Better strict-heldout F1 than raw features and Presto on the current reproduced protocol.
- Stronger representation value under geographic/dataset shift than raw flattened features.
- A stricter agricultural transfer protocol than many published clean/random-split comparisons.
- Evidence that target-domain unlabeled exposure is not carrying the grouped-holdout result, because strict heldout was similar to grouped holdout.

What would strengthen the main claim:
- Keep the OlmoEarth stress-sensitivity guard in the runner so identical stress embeddings cannot silently enter future tables.
- Add calibrated thresholds because Presto's AUROC remains stronger.
- Report few-shot curves across label budgets.
- Add variance across seeds or heldout datasets.
- Add a second benchmark or one larger SSL4EO-style pretraining run if time and compute allow.

Best venue framing:
- Workshop: strict heldout transfer, Presto comparison, robustness motivation, and LEM diagnostic are enough for a focused preliminary story.
- ICLR-style main submission: frame the project as a careful empirical study of self-supervised representation learning for irregular multimodal time series under domain shift. Agriculture is the testbed; the broader lesson is about temporal JEPA, missingness, target-domain exclusion, calibration, and raw-cue preservation.
- ICML-style main submission: only becomes natural if the work surfaces a more general method insight about objectives for irregular multimodal time series, beyond agricultural EO.

Claim-to-fame target:

> Compute-efficient crop-temporal representations that are more reliable under strict geographic transfer and real agricultural missingness than direct crop time-series baselines, and that clarify why broad EO embeddings can fail when simple red-edge/NIR/NDVI cues matter.

## Parking-Lot Links

Uncertainty / calibration:
- WR-CP: https://arxiv.org/abs/2501.13430
- Non-exchangeable conformal prediction + OT: https://arxiv.org/abs/2507.10425
- Multi-source conformal inference: https://arxiv.org/abs/2405.09331
- Conformal uncertainty quantification in EO: https://www.nature.com/articles/s41598-024-65954-w

Additional model links from the old CSV:
- Mamba-2 / structured state-space duality: https://arxiv.org/abs/2405.21060
  - Context: not JEPA-specific, but relevant if long Sentinel sequences make quadratic attention expensive. The source connects Transformers and SSMs and reports Mamba-2 as a faster selective-SSM architecture.
- WorldCereal deployment lessons: https://openreview.net/forum?id=eHW9HWitP0
  - Context: operational crop mapping with geospatial foundation models. Uses Presto in a WorldCereal case study and emphasizes domain requirements, adaptation to operational data, and rigorous empirical testing beyond standardized benchmark tasks.

## Evidence Handling

When adding benchmark entries:
- Prefer primary sources, official OpenReview pages, arXiv pages, official GitHub/model cards, or dataset cards.
- Include compute notes when disclosed.
- If a number comes from another comparison table rather than the original source, say so.
- Do not record vague leaderboard language without the benchmark, metric, split, and source.

## Deleted Source CSV Audit

`source_links_2026-04-10.csv` was deleted after its links were redistributed:
- Competitor/model links: `notes/competitors.md`
- JEPA and missing-modality method links: `notes/jepa.md`
- Benchmark, dataset, data access, and download links: `notes/benchmarks.md`
- Uncertainty and loose parking-lot links: this file

Direct PDF links from the CSV are kept in the relevant source lists even when they duplicate arXiv/OpenReview landing pages, because they explain where table extraction came from.

## Experiment Handoffs

### 2026-05-14 JEPA v1 Screen

Launched the CropHarvest JEPA v1 screening matrix on `digital-ag`.

- Output root: `/tmp/surf_runs/cropharvest_jepa_v1_screen`
- Jobs: `25`
- Worker PIDs: GPU `0` = `266070`, GPU `1` = `266071`
- First GPU 0 job: `v0_repro_seed11`
- First GPU 1 job: `v0_low_lr_seed42`
- Added runner flag: `--sample-s2-dropout-p`

The sample-level S2 outage augmentation drops Sentinel-2 for an entire training sequence when selected. This was added because the pilot weakness was `s2_off_tdrop50`, where the evaluation removes optical data globally and applies temporal drop.

Monitor:

```bash
ssh digital-ag 'nvidia-smi'
ssh digital-ag 'tail -f /tmp/surf_runs/cropharvest_jepa_v1_screen/worker_gpu0.log'
ssh digital-ag 'tail -f /tmp/surf_runs/cropharvest_jepa_v1_screen/worker_gpu1.log'
ssh digital-ag 'find /tmp/surf_runs/cropharvest_jepa_v1_screen -name exit_code -exec sh -c "printf \"%s \" \"$1\"; cat \"$1\"" sh {} \;'
```

After completion, copy back `/tmp/surf_runs/cropharvest_jepa_v1_screen` and rank configs by mean F1 over `clean`, `sensor_off_s2`, `temporal_drop_50`, `temporal_drop_70`, and `s2_off_tdrop50`. Tie-break on AUROC under `s2_off_tdrop50`.

### 2026-05-14 JEPA v1 Results

Pulled the completed v1 screen from `digital-ag`:

- Remote: `/tmp/surf_runs/cropharvest_jepa_v1_screen`
- Local: `artifacts/runs/cropharvest_jepa_v1_screen`
- Runs: `25 / 25`
- Exit codes: all `0`

Analysis CSVs were written to:

- `artifacts/runs/cropharvest_jepa_v1_screen/analysis/config_ranking.csv`
- `artifacts/runs/cropharvest_jepa_v1_screen/analysis/priority_condition_f1.csv`
- `artifacts/runs/cropharvest_jepa_v1_screen/analysis/condition_summary.csv`
- `artifacts/runs/cropharvest_jepa_v1_screen/analysis/budget100_summary.csv`
- `artifacts/runs/cropharvest_jepa_v1_screen/analysis/train_summary_by_config.csv`

Top ranking by the requested priority F1:

1. `medium_low_lr`: `0.7825`
2. `v0_repro`: `0.7776`, but only one seed
3. `hard_robust_low_lr`: `0.7757`
4. `v0_low_lr`: `0.7755`
5. `full_s2_outage_robust`: `0.7754`

Main read:

- Promote `medium_low_lr`; it gets clean 100% F1 to `0.8234`.
- Robust augmentation matters for `s2_off_tdrop50`; `no_robust` falls to `0.7358` mean F1 and `0.7217` AUROC.
- `full_s2_outage_robust` did not win as implemented. Try it next combined with timestep S2 blackout rather than replacing it.
- DOY should stay; `no_doy` was the weakest priority score.
- Keep using early stopping/checkpoint selection. Best epochs were mostly `4-11`.

### 2026-05-14 Results Back: Next Experiment Read

Read `notes/experiments.md` and the returned artifacts under `artifacts/runs/cropharvest_jepa_v1_screen`.

Main conclusions:

- `medium_low_lr` is the current strongest config. It wins the three-seed priority score and has the best clean full-label result.
- Robustness augmentation is justified by the combined S2-off plus temporal-drop condition. Clean scores are similar without it, but severe-stress AUROC/F1 drop noticeably.
- Day-of-year should stay. The ablation is modest but consistently weaker.
- The exact sample-level S2 outage variant did not win. The higher-signal follow-up is a combined whole-sequence S2 outage plus timestep S2 blackout setting on the medium model.
- Best validation checkpoints arrive early, usually epoch 4-11. Use max 30 epochs with patience around 5, or keep 20 epochs if early stopping is not wired yet.
- The next evaluation bottleneck is split design. Random stratified results are encouraging, but the next run should add dataset-held-out probes using the `dataset` property in `labels.geojson`.

Recommended next run:
- `cropharvest_jepa_v2_confirm_generalization`
- Confirm medium configs over five seeds.
- Add medium hard/dual-S2 robustness variants.
- Add one larger model as a scale probe.
- Evaluate both random split and dataset-held-out probe splits, especially `togo-eval`, `rwanda-ceo`, `togo`, `ethiopia`, and `lem-brazil` where both classes exist.

### 2026-05-14 JEPA v2 Results Read

Read the completed `cropharvest_jepa_v2_confirm_generalization` run.

Main results:

- New v2 runs completed: `24 / 24`.
- Best random-split config: `large_dual_s2`, priority F1 `0.7869`, clean 100% F1 `0.8289`, clean AUROC `0.8764`.
- Second random-split config: `large_default`, priority F1 `0.7861`, clean 100% F1 `0.8257`.
- Best grouped-holdout config: `large_dual_s2`, grouped F1 `0.5078`, grouped AUROC `0.6889`.
- `large_default` has nearly identical grouped performance and slightly higher grouped AUROC: `0.6891`.
- Medium configs are similar, but large configs are consistently ahead on random split and grouped holdout.
- Dual S2 masking is a small positive, not a dramatic result. Keep it, but do not make it the main contribution.
- Robustness still matters most under `s2_off_tdrop50`; compared with `medium_no_robust`, `large_dual_s2` gains about `+0.0091` F1 and `+0.0251` AUROC on the combined stress condition.
- Grouped holdout is much harder than random split. The weak heldouts are `lem-brazil` and `rwanda-ceo`; `ethiopia` is relatively easy.
- Raw grouped baseline was only computed for one run, but showed a large F1 gap and smaller AUROC gap: raw grouped F1 `0.2695`, AUROC `0.6445`; JEPA seed-42 medium default grouped F1 `0.4890`, AUROC `0.6741`.

Recommended next work:

- Treat `large_dual_s2` as the current strongest config.
- Run `large_dual_s2` seeds `13` and `17` to match five-seed medium evidence.
- Run grouped raw baselines more carefully and report AUROC/balanced accuracy alongside F1 because grouped F1 is sensitive to threshold calibration.
- Highest-gradient next direction: external baseline reproduction or a stricter transfer protocol, not another CropHarvest random-split sweep.
- Keep architecture tuning only when it tests a specific failure mode observed in the current results.

### 2026-05-15 JEPA v3 Strict Dataset-Heldout Read

Read the completed `cropharvest_jepa_v3_strict_dataset_holdout` run.

Main correction:

- Strict heldout drops hard relative to random split, but not relative to the previous clean grouped-holdout probe. Earlier wording that target-domain unlabeled SSL exposure was driving the grouped result was too strong.

Main results:

- Runs completed: `30 / 30`.
- `large_dual_s2` strict priority F1: `0.4941`; AUROC `0.6703`; balanced accuracy `0.6093`.
- `large_default` strict priority F1: `0.4898`; AUROC `0.6714`; balanced accuracy `0.6088`.
- Raw flattened strict priority F1: `0.3112`; AUROC `0.6111`; balanced accuracy `0.5437`.
- JEPA beats raw overall and under every aggregate stress condition.
- LEM Brazil is the failure case: raw beats JEPA on strict priority F1 and AUROC. This is the next diagnostic target.

Next highest-gradient direction:

- Reproduce the strongest practical external baseline on the same strict heldout protocol, with OlmoEarth first if setup is reasonable and Presto second.
- Diagnose LEM Brazil before adding more architecture variants.
- Track AUROC and balanced accuracy alongside F1 because threshold behavior is unstable under dataset shift.

### 2026-05-23 Repair Pass for Experiments [1]-[5]

Pulled the repaired result tree back from `digital-ag` into `artifacts/runs/repaired_experiments`.

What was fixed:

- Added source-validation threshold calibration to probe evaluation.
- Added raw-stat-only and embedding-plus-raw-stat probes for JEPA embeddings.
- Added sample-level S2 dropout in the repaired JEPA training path.
- Added dry-runs before the main runner and Presto stress jobs.
- Reran strict heldout with target datasets excluded from SSL pretraining and probe training.
- Reran Presto stress conditions and later reran fixed OlmoEarth stress extraction in `artifacts/runs/repaired_external_v3`; OlmoEarth stress metrics now change across conditions.
- Reran LEM Brazil diagnostics after the strict heldout repair pass.

Before/after read:

| Experiment | Before fixing issues | After repair | Interpretation |
|---|---:|---:|---|
| `[1]` v0 random split | JEPA priority F1 `0.7718`, AUROC `0.7874` | JEPA priority F1 `0.7734`, AUROC `0.7880`; hybrid JEPA+raw-stats F1 `0.7831`, AUROC `0.7952`; calibrated JEPA F1 `0.8338` | Original signal holds. Raw stats are useful as a hybrid probe. |
| `[2]` v1 screen | best `medium_low_lr` F1 `0.7825` | best `medium_low_lr` F1 `0.7755`, AUROC `0.7899`; calibrated F1 `0.8336` | Ranking compresses after repair, but `medium_low_lr` remains the best random-split config. |
| `[3]` grouped holdout | best previous grouped config `large_dual_s2` F1 `0.5078`, AUROC `0.6889` | best repaired medium-screen grouped config `no_doy` F1 `0.4959`, AUROC `0.6525`; best calibrated F1 among repaired grouped configs is `medium_low_lr` at `0.5814` | Grouped transfer remains hard. The repaired grouped pass is not one-to-one with every old large-config run. |
| `[4]` strict heldout | `large_dual_s2` F1 `0.4941`, AUROC `0.6703`; `large_default` F1 `0.4898`, AUROC `0.6714` | `large_dual_s2` F1 `0.4917`, AUROC `0.6655`, calibrated F1 `0.5839`; `large_default` F1 `0.4877`, AUROC `0.6674`, calibrated F1 `0.5854` | Strict JEPA result essentially holds. Calibration raises F1 substantially; `large_dual_s2` keeps best default F1, while `large_default` slightly leads AUROC/calibrated F1. |
| `[5]` external baselines | OlmoEarth F1 `0.6006`, AUROC `0.7355`; Presto F1 `0.3897`, AUROC `0.7289` | OlmoEarth priority F1 `0.5553`, AUROC `0.6818`, calibrated F1 `0.5782`; Presto priority F1 `0.3845`, AUROC `0.7273`, calibrated F1 `0.5756` | OlmoEarth stress numbers are now usable after the fixed rerun. OlmoEarth leads by default priority F1; Presto leads by AUROC. |

Detailed per-metric before/after:

| Experiment | Comparison | Metric | Before | After | Delta |
|---|---:|---:|---:|---:|---:|
| `[1]` | raw_flattened | F1 | 0.7579 | 0.7599 | +0.0020 |
| `[1]` | raw_flattened | AUROC | 0.7547 | 0.7527 | −0.0020 |
| `[1]` | raw_flattened | calibrated F1 | — | 0.8266 | — |
| `[1]` | surf_jepa_v0 | F1 | 0.7718 | 0.7734 | +0.0016 |
| `[1]` | surf_jepa_v0 | AUROC | 0.7874 | 0.7880 | +0.0006 |
| `[1]` | surf_jepa_v0 | calibrated F1 | — | 0.8338 | — |
| `[2]` | hard_robust_low_lr | priority F1 | 0.7757 | 0.7743 | −0.0013 |
| `[2]` | medium_low_lr | priority F1 | 0.7825 | 0.7755 | −0.0070 |
| `[2]` | medium_low_lr | calibrated F1 | — | 0.8336 | — |
| `[2]` | no_doy | priority F1 | 0.7729 | 0.7740 | +0.0011 |
| `[4]` | large_default | strict F1 | 0.3112 | 0.4877 | +0.1765 |
| `[4]` | large_default | strict AUROC | 0.6111 | 0.6674 | +0.0562 |
| `[4]` | large_default | calibrated F1 | — | 0.5854 | — |
| `[4]` | large_dual_s2 | strict F1 | 0.3112 | 0.4917 | +0.1806 |
| `[4]` | large_dual_s2 | strict AUROC | 0.6111 | 0.6655 | +0.0544 |
| `[4]` | large_dual_s2 | calibrated F1 | — | 0.5839 | — |
| `[5]` | olmoearth | priority F1 | 0.6006 | 0.5995 | −0.0010 |
| `[5]` | olmoearth | priority AUROC | 0.7355 | 0.7296 | −0.0058 |
| `[5]` | olmoearth | calibrated F1 | — | 0.6125 | — |
| `[5]` | presto | priority F1 | 0.3897 | 0.3845 | −0.0052 |
| `[5]` | presto | priority AUROC | 0.7289 | 0.7273 | −0.0015 |
| `[5]` | presto | calibrated F1 | — | 0.5756 | — |

Keep `notes/experiments.md` as the canonical current experiment log with only the trusted repaired version of each experiment. Keep this note as the audit trail for what changed during the repair pass.

### 2026-05-24 Archived Raw-Cue Long Run

This raw-cue long run is no longer part of the canonical experiment sequence. Treat it as an archived failed/caveated attempt, not as experiment `[6]`. The active experiment log should skip from `[5]` to the next clean experiment number.

What ran:

| Arm | Intended purpose |
|---|---|
| `surf_v4_rawcue_control` | JEPA plus a raw-cue prediction loss. |
| `surf_v4_rawcue_olmo_distill` | JEPA plus raw-cue prediction plus OlmoEarth teacher alignment. |

Intended objectives:

```text
control: L = L_JEPA + 0.05 * L_rawcue
teacher: L = L_JEPA + 0.05 * L_rawcue + 0.10 * L_olmoearth
```

Final checkpoint read:

| Arm | Priority F1 | Priority AUROC | Clean 100% F1 | Clean 100% AUROC | LEM Priority F1 | LEM Priority AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `surf_v4_rawcue_control` | `0.4833` | `0.6691` | `0.4997` | `0.7445` | `0.2970` | `0.5199` |
| `surf_v4_rawcue_olmo_distill` | `0.4915` | `0.6692` | `0.5752` | `0.7741` | `0.3022` | `0.5128` |

Checkpoint probes suggested earlier checkpoints were better:

| Arm | Best clean-100 F1 | Best clean-100 AUROC | Final clean-100 F1 | Final clean-100 AUROC |
|---|---:|---:|---:|---:|
| `surf_v4_rawcue_control` | `0.5604` at step `101023` | `0.7802` at step `101023` | `0.4997` | `0.7445` |
| `surf_v4_rawcue_olmo_distill` | `0.6058` at step `55260` | `0.7889` at step `55260` | `0.5752` | `0.7741` |

Known issues with the run:

- It used only CropHarvest source data for SSL, not SSL4EO-S12 or another larger unlabeled pool.
- The teacher arm did not use cached OlmoEarth targets and paid the teacher forward pass every step.
- Final full evaluation was embedding-only, with no raw-stats-only or embedding-plus-raw-stats probes.
- Threshold calibration was not implemented despite being part of the planned evaluation.
- The final full evaluation used final checkpoints even though checkpoint probes showed earlier checkpoints were better.
- The resulting numbers are not trustworthy enough for the canonical experiment log.

Repository cleanup:

- Removed the active runner `runners/[6].py`.
- Removed the active notebook dashboard `notebooks/[6].ipynb`.
- Removed README/notebook-guide references to `[6]`.
- Do not resurrect this exact code path. If raw-cue preservation comes back, make it a new clean experiment with a larger SSL pool, cached teacher targets, calibrated probes, and best-checkpoint final evaluation from the start.

## New [6] Plan: Strict Hybrid Raw-Cue Probe

The canonical `[6]` is now a probe-only strict hybrid experiment, not the old raw-cue training/distillation run.

Goal:

> Test whether the gap to OlmoEarth/Presto comes from weak JEPA representations or from JEPA embeddings discarding simple red-edge/NIR/NDVI magnitude cues that still transfer under strict heldout evaluation.

Runner added:

- `runners/[6].py`
- `notebooks/[6].ipynb`

Protocol:

- Load existing repaired `[4]` strict checkpoints from `artifacts/runs/repaired_experiments/repair_4_strict`.
- Use configs `large_default` and `large_dual_s2`.
- Use seeds `7`, `11`, and `42`.
- Use heldouts `rwanda-ceo`, `togo`, `togo-eval`, `ethiopia`, and `lem-brazil`.
- Use conditions `clean`, `sensor_off_s2`, `temporal_drop_50`, `temporal_drop_70`, and `s2_off_tdrop50`.
- Enforce `strict checkpoint holdout == evaluation holdout`; do not sweep every checkpoint over every holdout.
- Evaluate `surf_jepa_v0`, `raw_stats`, `surf_jepa_v0_plus_raw_stats`, and `raw_flattened`.

This should run before any new raw-cue objective, teacher distillation, or larger-scale pretraining.

`notebooks/[6].ipynb` is the visualization dashboard for the completed probe. It should read the table-grade artifact at `artifacts/runs/strict_hybrid_raw_cue_probe_tablegrade`, plot the priority scorecard, hybrid-minus-embedding deltas, per-heldout heatmaps, LEM Brazil breakdowns, label-budget curves, and optional `[5]` external-baseline context.

### 2026-05-25 Runner Independence And [6] Probe Cleanup

Runners must not import or execute other runners. Shared work now lives in `src/evaluation.py`, and `[2]`, `[3]`, `[4]`, and `[6]` import normal library functions instead of loading `[1].py`.

The shared probe path now uses stable logistic-regression settings for table-grade reruns:

- solver: `liblinear`
- max iterations: `20000`
- tolerance: `1e-5`
- per-row convergence metadata: `probe_converged`, `probe_convergence_warnings`, `probe_n_iter`, and solver settings

`[6]` summaries now also include convergence audit fields. The table-grade rerun in `artifacts/runs/strict_hybrid_raw_cue_probe_tablegrade` completed with zero convergence warnings over `3000` probe rows.

Key `[6]` result:

- Best embedding-only JEPA: `large_dual_s2 / surf_jepa_v0`, priority F1 `0.4915`, AUROC `0.6655`, calibrated F1 `0.5839`.
- Best hybrid: `large_dual_s2 / surf_jepa_v0_plus_raw_stats`, priority F1 `0.5307`, AUROC `0.6939`, calibrated F1 `0.5911`.
- Hybrid delta over embedding-only: `+0.0392` priority F1 and `+0.0284` AUROC.
- Hybrid beats the repaired OlmoEarth baseline by AUROC and calibrated F1, but not by default priority F1.
- Hybrid beats the repaired Presto baseline by default priority F1 and calibrated F1, but not by AUROC.

Interpretation: the learned embedding is missing useful raw red-edge/NIR/NDVI/SAR magnitude information. The next model-design experiment should preserve those cues inside the encoder instead of relying on probe-time concatenation.

## [7] Temporal Block-JEPA v1

`[7]` is now staged as the first real architecture experiment. The bounded first screen should pretrain on a packaged SSL4EO-S12 v1.1 subset and use CropHarvest only as the strict heldout probe/evaluation dataset.

Files:

- `src/jepa.py`: adds `TemporalBlockJepaModel`, `TemporalBlockPredictor`, and masked latent cosine loss.
- `runners/[7].py`: strict heldout architecture runner with arms `A_control`, `B_full_target`, `C_transformer_predictor`, `D_multiblock`, `E_cross_modal`, `F_rawcue`, and `G_full`.

Important protocol choices:

- Pretrain on `data/processed/ssl4eo_s12_v11_48k.zarr` for the bounded first architecture screen. This is `49,152` SSL4EO samples, exactly `768` shards, with a separate stream-and-evict cache path capped at `60 GiB`. This is a packaged SSL4EO screen, not a true agriculture-masked corpus.
- Evaluate/probe on `data/cropharvest/processed/v2.zarr`; do not pretrain on CropHarvest in `[7]`.
- The pretrain and eval zarr stores must expose exact matching `s2_bands`, `s1_bands`, and `climate_bands`; counts alone are not acceptable.
- Temporal length and spatial patch size may differ: SSL4EO should be `T=4`, `H=W=16`; CropHarvest eval can remain monthly point/pseudo-patch data.
- Default holdouts: `rwanda-ceo`, `togo`, `togo-eval`, `ethiopia`, `lem-brazil`.
- Default priority conditions: `clean`, `sensor_off_s2`, `temporal_drop_50`, `temporal_drop_70`, `s2_off_tdrop50`.
- Default seed list is only `42` to keep first launch sane; add more seeds explicitly.
- Use `--smoke-model` for synthetic forward/loss checks without touching the CropHarvest files.

Do not treat `[7]` as a second-benchmark experiment or a custom raw-Sentinel chip pipeline. It answers whether the current next-step JEPA was too weak architecturally when moved from CropHarvest-only pretraining to packaged patch-time-series SSL4EO pretraining.

Pre-launch fixes made after review:

- `TemporalBlockJepaModel` keeps the EMA target encoder in eval mode even while the online model is training.
- B/C-style non-multiblock arms now predict one short target block, not every future timestep.
- Validation target masks are deterministic, so best-checkpoint selection is not driven by fresh random masks every epoch.
- `src/cropharvest.py` now builds the v2 channel contract for `cropharvest_v2.zarr`.
- `[7]` asserts exact `s2_bands`, `s1_bands`, and `climate_bands`; counts alone are not enough.
- Raw-cue targets now average over spatial pixels before temporal statistics, so they work for SSL4EO patches and CropHarvest point tensors.
- `[7]` pretrains once per arm/seed and evaluates all requested CropHarvest holdouts from that checkpoint.
- Synthetic `--smoke-model` checks passed for all arms.

First launch should be the two-holdout screen from `notes/experiments.md`, not the full all-holdout matrix.

## Source Links Inventory (archived 2026-04-12)

Ranked literature and data sources gathered during project startup. Tier: P0=core, P1=strong, P2=useful, P3=context, P4=archive.

### P0–P1 (high priority)

| R | Tier | Score | Group | Notes | URL |
|---|---|---|---|---|---|
| 1 | P0 | 100 | core_ml | I-JEPA | https://arxiv.org/abs/2301.08243 |
| 2 | P0 | 100 | core_ml | Presto arXiv version | https://arxiv.org/abs/2304.14065 |
| 3 | P0 | 100 | competitor_models | Galileo | https://arxiv.org/abs/2502.09356 |
| 4 | P0 | 100 | core_ml | V-JEPA 2 | https://arxiv.org/abs/2506.09985 |
| 5 | P0 | 100 | competitor_models | Presto OpenReview | https://openreview.net/forum?id=Iip7rt9UL3 |
| 6 | P0 | 100 | competitor_models | Presto PDF | https://openreview.net/pdf?id=Iip7rt9UL3 |
| 7 | P0 | 99 | core_ml | X-JEPA WACV 2026 | https://openaccess.thecvf.com/content/WACV2026/papers/Choudhury_X-JEPA...pdf |
| 8 | P0 | 98 | competitor_models | Copernicus-FM | https://arxiv.org/abs/2503.11849 |
| 9 | P0 | 98 | competitor_models | TerraFM | https://arxiv.org/abs/2506.06281 |
| 10 | P0 | 95 | pretraining_datasets | SSL4EO-S12 paper | https://arxiv.org/abs/2211.07044 |
| 11 | P0 | 95 | benchmarks | CropHarvest paper PDF | https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/54229abfcfa5649e7003b83dd4755294-Paper-round2.pdf |
| 12 | P0 | 93 | core_ml | Neural CDE | https://arxiv.org/abs/2005.08926 |
| 13 | P0 | 93 | core_ml | MiDl missing modality | https://arxiv.org/abs/2404.15161 |
| 14 | P0 | 93 | core_ml | Timer-XL | https://arxiv.org/abs/2410.04803 |
| 15 | P0 | 93 | core_ml | RoMA | https://arxiv.org/abs/2503.10392 |
| 16 | P0 | 93 | core_ml | SGMA incomplete multimodal RS | https://arxiv.org/abs/2603.02505 |
| 17 | P0 | 93 | core_ml | Timer-S1 | https://arxiv.org/abs/2603.04791 |
| 18 | P1 | 91 | benchmarks | BigEarthNet HDF5 (downloaded mini val chunk) | https://huggingface.co/datasets/lc-col/bigearthnet |
| 19 | P1 | 90 | data_access | SSL4EO-S12 v1.1 paper | https://arxiv.org/abs/2503.00168 |
| 20 | P1 | 90 | core_ml | MD2N ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_...pdf |
| 21 | P1 | 90 | benchmarks | CropHarvest Zenodo record | https://zenodo.org/records/5037916 |
| 22 | P1 | 89 | adjacent_models | Prithvi-EO-2.0 | https://arxiv.org/abs/2412.02732 |
| 23 | P1 | 89 | adjacent_models | AnySat | https://arxiv.org/abs/2412.14123 |
| 24 | P1 | 89 | adjacent_models | TerraMind | https://arxiv.org/abs/2504.11171 |
| 25 | P1 | 89 | adjacent_models | THOR | https://arxiv.org/abs/2601.16011 |
| 26 | P1 | 87 | ag_eval | Generalizability of FMs for crop type mapping | https://arxiv.org/abs/2409.09451 |
| 27 | P1 | 87 | ag_eval | Deploying GeoFMs in WorldCereal | https://arxiv.org/abs/2508.00858 |
| 28 | P1 | 87 | ag_eval | Harvesting AlphaEarth | https://arxiv.org/abs/2601.00857 |
| 29 | P1 | 87 | data_access | CropHarvest GitHub | https://github.com/nasaharvest/cropharvest |
| 30 | P1 | 87 | data_access | SSL4EO-S12 v1.1 dataset card | https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1 |
| 31 | P1 | 86 | benchmarks | EuroSAT (torchvision download) | https://github.com/phelber/eurosat |
| 32 | P1 | 85 | ag_benchmarks | EuroCropsML | https://arxiv.org/abs/2407.17458 |
| 33 | P1 | 85 | ag_benchmarks | Fields of The World | https://arxiv.org/abs/2409.16252 |

### P2 (useful)

| R | Tier | Score | Group | Notes | URL |
|---|---|---|---|---|---|
| 34 | P2 | 83 | benchmarks | PASTIS (too large for local disk) | https://huggingface.co/datasets/GFM-Bench/PASTIS |
| 35 | P2 | 83 | benchmarks | PASTIS-HD listing | https://huggingface.co/datasets/IGNF/PASTIS-HD |
| 36 | P2 | 82 | benchmarks | BigEarthNet alt mirror | https://huggingface.co/datasets/GFM-Bench/BigEarthNet |
| 37 | P2 | 81 | data_access | GEE Sentinel-1 GRD | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD |
| 38 | P2 | 81 | data_access | GEE Sentinel-2 SR Harmonized | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED |
| 39 | P2 | 81 | data_access | Sentinel legal notice | https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice |
| 40 | P2 | 78 | benchmarks | EuroCrops (no hosted data) | https://huggingface.co/datasets/maja601/EuroCrops |
| 41 | P2 | 75 | data_access | QuickStats metadata | https://catalog.data.gov/dataset/quick-stats-agricultural-database-api |
| 42 | P2 | 75 | data_access | AgERA5 CDS dataset | https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators |
| 43 | P2 | 75 | data_access | AgERA5 documentation | https://confluence.ecmwf.int/pages/viewpage.action?pageId=278551004 |
| 44 | P2 | 75 | data_access | Copernicus Data Space | https://dataspace.copernicus.eu/ |
| 45 | P2 | 75 | data_access | GEE ERA5 daily | https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY |
| 46 | P2 | 75 | data_access | GEE ERA5-Land daily aggregated | https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR |
| 47 | P2 | 75 | data_access | GEE ERA5-Land hourly | https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY |
| 48 | P2 | 75 | data_access | MOD13Q1 NDVI/EVI | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1 |
| 49 | P2 | 75 | data_access | MOD15A2H LAI/FPAR | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD15A2H |
| 50 | P2 | 75 | data_access | SMAP L4 soil moisture | https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008 |
| 51 | P2 | 75 | data_access | USDA NASS QuickStats API | https://quickstats.nass.usda.gov/api |

### P3–P4 (context / archive)

| R | Tier | Score | Group | Notes | URL |
|---|---|---|---|---|---|
| 52 | P3 | 71 | uncertainty | Multi-Source Conformal Inference | https://arxiv.org/abs/2405.09331 |
| 53 | P3 | 71 | uncertainty | WR-CP | https://arxiv.org/abs/2501.13430 |
| 54 | P3 | 71 | uncertainty | Non-exchangeable CP + OT | https://arxiv.org/abs/2507.10425 |
| 55 | P3 | 68 | additional_ml | Opened during source exploration (Mamba-2 SSM) | https://arxiv.org/abs/2405.21060 |
| 56 | P3 | 68 | additional_ml | Opened during source exploration (WorldCereal) | https://openreview.net/forum?id=eHW9HWitP0 |
| 57 | P3 | 68 | uncertainty | Conformal UQ in EO | https://www.nature.com/articles/s41598-024-65954-w |
| 58 | P4 | 62 | code_reference | CropHarvest bands.py (band ordering) | https://raw.githubusercontent.com/nasaharvest/cropharvest/main/cropharvest/bands.py |
| 59 | P4 | 59 | downloads | features.tar.gz (downloaded) | https://zenodo.org/records/5037916/files/features.tar.gz?download=1 |
| 60 | P4 | 59 | downloads | labels.geojson (downloaded) | https://zenodo.org/records/5037916/files/labels.geojson?download=1 |
| 61 | P4 | 55 | downloads | features.tar.gz (Zenodo page) | https://zenodo.org/records/5037916/files/features.tar.gz |
| 62 | P4 | 55 | downloads | labels.geojson (Zenodo page) | https://zenodo.org/records/5037916/files/labels.geojson |
