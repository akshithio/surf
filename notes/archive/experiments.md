# Experiment Log

This file logs executed experiments, commands, and key outputs.

## 2026-04-10

### Backfill: Earlier successful pipeline checks
- Built synthetic multimodal raster sample and verified end-to-end dataset build to Zarr.
- Ran smoke pretraining on synthetic Zarr.
- Downloaded and unpacked CropHarvest from Zenodo.
- Converted CropHarvest subset (500 samples) to multimodal Zarr and ran pretraining.

### Backfill: Completed mini ablation on CropHarvest-500
- Output summary CSV: `artifacts/figures/exp_cropharvest_500_summary.csv`
- Runs:
  - `baseline_d0`
  - `dropout_d02`
  - `dropout_d05`
  - `modeldim_128`

### Next block requested by user
- Plan:
  1. Build CropHarvest 5k Zarr subset.
  2. Train next-embedding baseline on 5k subset.
  3. Evaluate sensor-off robustness (`s2`, `s1`, `climate`).
  4. Evaluate irregular sampling robustness (drop 30%, 50% timesteps).


### 2026-04-10 13:54:06 EDT - baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000/train_modeldim128_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.0007137837674235925,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 13:54:10 EDT - sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000/train_modeldim128_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.002209724683780223,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 13:54:14 EDT - sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000/train_modeldim128_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.0008675716526340693,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 13:54:18 EDT - sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000/train_modeldim128_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.0010334758408134803,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 13:54:21 EDT - temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000/train_modeldim128_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\```json
{
  "val_loss": 0.0029081152169965208,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\```

### 2026-04-10 13:54:25 EDT - temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000/train_modeldim128_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\```json
{
  "val_loss": 0.004674740950576961,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\```

### 2026-04-10 5k Pretrain Block (full run)
- Command: PYTHONPATH=src .venv/bin/python -m surf.cli build-cropharvest-zarr --arrays-dir data/raw/cropharvest/features/arrays --output-zarr /tmp/cropharvest_5000.zarr --max-samples 5000
- Command: PYTHONPATH=src .venv/bin/python -m surf.cli inspect-zarr --zarr-path /tmp/cropharvest_5000.zarr
- Command: PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_cropharvest_5000/train_modeldim128_d02 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2
- Output artifact: /tmp/exp_cropharvest_5000/train_modeldim128_d02/best_checkpoint.pt
- Output artifact: /tmp/exp_cropharvest_5000/train_modeldim128_d02/train_history.json
- Key result: best_val_loss=0.0007137835054891184
- Train wall time (sec): 139.06938695907593
- Robustness summary CSV: artifacts/figures/exp_cropharvest_5000_robustness.csv

### 2026-04-10 14:05:12 EDT - control_d0_baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.000767585908761248,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:05:16 EDT - control_d0_sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.23535709828138351,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:05:19 EDT - control_d0_sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0010809956002049148,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:05:22 EDT - control_d0_sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0025578041095286608,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:05:25 EDT - control_d0_temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
```json
{
  "val_loss": 0.06659940257668495,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
```

### 2026-04-10 14:05:29 EDT - control_d0_temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
```json
{
  "val_loss": 0.11330930888652802,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
```

### 2026-04-10 5k Control Pretrain Block (no modality dropout)
- Command: PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.0
- Output artifact: /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/best_checkpoint.pt
- Output artifact: /tmp/exp_cropharvest_5000_control_d0/train_modeldim128_d0/train_history.json
- Key result: best_val_loss=0.0007675855595152825
- Train wall time (sec): 132.81257915496826
- Robustness summary CSV: artifacts/figures/exp_cropharvest_5000_control_d0_robustness.csv

### 2026-04-10 14:21:14 EDT - seed_repl seed=7 p=0.0 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_seed_repl_5000/seed_7_d0 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.0 --seed 7`
- Output artifact: /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt
- Output artifact: /tmp/exp_seed_repl_5000/seed_7_d0/train_history.json
- best_val_loss: 0.0007926784746814519
- train_seconds: 137.65278100967407

### 2026-04-10 14:21:19 EDT - seed_repl seed=7 p=0.0 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0007695441454416141,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:21:22 EDT - seed_repl seed=7 p=0.0 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.1852196678519249,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:21:26 EDT - seed_repl seed=7 p=0.0 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0013564634427893907,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:21:30 EDT - seed_repl seed=7 p=0.0 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.003548976790625602,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:21:34 EDT - seed_repl seed=7 p=0.0 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
```json
{
  "val_loss": 0.05985820572823286,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
```

### 2026-04-10 14:21:38 EDT - seed_repl seed=7 p=0.0 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
```json
{
  "val_loss": 0.10123267211019993,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
```

### 2026-04-10 14:24:00 EDT - seed_repl seed=7 p=0.2 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_seed_repl_5000/seed_7_d02 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --seed 7`
- Output artifact: /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt
- Output artifact: /tmp/exp_seed_repl_5000/seed_7_d02/train_history.json
- best_val_loss: 0.000688613363308832
- train_seconds: 140.21289587020874

### 2026-04-10 14:24:05 EDT - seed_repl seed=7 p=0.2 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0006743733829353005,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:24:08 EDT - seed_repl seed=7 p=0.2 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.004130107467062771,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:24:12 EDT - seed_repl seed=7 p=0.2 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.000926966589759104,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:24:16 EDT - seed_repl seed=7 p=0.2 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0014417824859265238,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:24:20 EDT - seed_repl seed=7 p=0.2 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
```json
{
  "val_loss": 0.0038623601431027055,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
```

### 2026-04-10 14:24:24 EDT - seed_repl seed=7 p=0.2 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
```json
{
  "val_loss": 0.006312897894531488,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
```

### 2026-04-10 14:26:47 EDT - seed_repl seed=42 p=0.0 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_seed_repl_5000/seed_42_d0 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.0 --seed 42`
- Output artifact: /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt
- Output artifact: /tmp/exp_seed_repl_5000/seed_42_d0/train_history.json
- best_val_loss: 0.0007675855595152825
- train_seconds: 140.68164992332458

### 2026-04-10 14:26:50 EDT - seed_repl seed=42 p=0.0 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.000767585908761248,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:27:41 EDT - seed_repl_2seed seed=7 p=0.0 pretrain/checkpoint
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_seed_repl_5000/seed_7_d0 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.0 --seed 7`
- best_val_loss: 0.0007926784746814519
- train_seconds: 137.65278100967407
- checkpoint: /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt

### 2026-04-10 14:27:45 EDT - seed_repl_2seed seed=7 p=0.0 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0007695441454416141,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:27:48 EDT - seed_repl_2seed seed=7 p=0.0 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.1852196678519249,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:27:52 EDT - seed_repl_2seed seed=7 p=0.0 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0013564634427893907,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:27:55 EDT - seed_repl_2seed seed=7 p=0.0 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.003548976790625602,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:27:59 EDT - seed_repl_2seed seed=7 p=0.0 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
```json
{
  "val_loss": 0.05985820572823286,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
```

### 2026-04-10 14:28:03 EDT - seed_repl_2seed seed=7 p=0.0 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
```json
{
  "val_loss": 0.10123267211019993,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
```

### 2026-04-10 14:28:03 EDT - seed_repl_2seed seed=7 p=0.2 pretrain/checkpoint
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_seed_repl_5000/seed_7_d02 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --seed 7`
- best_val_loss: 0.000688613363308832
- train_seconds: 140.21289587020874
- checkpoint: /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt

### 2026-04-10 14:28:07 EDT - seed_repl_2seed seed=7 p=0.2 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0006743733829353005,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:10 EDT - seed_repl_2seed seed=7 p=0.2 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.004130107467062771,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:14 EDT - seed_repl_2seed seed=7 p=0.2 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.000926966589759104,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:17 EDT - seed_repl_2seed seed=7 p=0.2 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0014417824859265238,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:21 EDT - seed_repl_2seed seed=7 p=0.2 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
```json
{
  "val_loss": 0.0038623601431027055,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
```

### 2026-04-10 14:28:24 EDT - seed_repl_2seed seed=7 p=0.2 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_7_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
```json
{
  "val_loss": 0.006312897894531488,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
```

### 2026-04-10 14:28:24 EDT - seed_repl_2seed seed=42 p=0.0 pretrain/checkpoint
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_seed_repl_5000/seed_42_d0 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.0 --seed 42`
- best_val_loss: 0.0007675855595152825
- train_seconds: 140.68164992332458
- checkpoint: /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt

### 2026-04-10 14:28:28 EDT - seed_repl_2seed seed=42 p=0.0 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.000767585908761248,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:31 EDT - seed_repl_2seed seed=42 p=0.0 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.23535709828138351,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:35 EDT - seed_repl_2seed seed=42 p=0.0 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0010809956002049148,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:39 EDT - seed_repl_2seed seed=42 p=0.0 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
```json
{
  "val_loss": 0.0025578041095286608,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
```

### 2026-04-10 14:28:42 EDT - seed_repl_2seed seed=42 p=0.0 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
```json
{
  "val_loss": 0.06659940257668495,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
```

### 2026-04-10 14:28:46 EDT - seed_repl_2seed seed=42 p=0.0 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d0/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
```json
{
  "val_loss": 0.11330930888652802,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
```

### 2026-04-10 14:31:45 EDT - seed_repl_2seed seed=42 p=0.2 pretrain/checkpoint
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_seed_repl_5000/seed_42_d02 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --seed 42`
- best_val_loss: 0.0007137835054891184
- train_seconds: 128.3029260635376
- checkpoint: /tmp/exp_seed_repl_5000/seed_42_d02/best_checkpoint.pt

### 2026-04-10 14:31:48 EDT - seed_repl_2seed seed=42 p=0.2 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.0007137837674235925,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 14:31:52 EDT - seed_repl_2seed seed=42 p=0.2 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.002209724683780223,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 14:31:55 EDT - seed_repl_2seed seed=42 p=0.2 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.0008675716526340693,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 14:31:59 EDT - seed_repl_2seed seed=42 p=0.2 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\```json
{
  "val_loss": 0.0010334758408134803,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\```

### 2026-04-10 14:32:03 EDT - seed_repl_2seed seed=42 p=0.2 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\```json
{
  "val_loss": 0.0029081152169965208,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\```

### 2026-04-10 14:32:06 EDT - seed_repl_2seed seed=42 p=0.2 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_seed_repl_5000/seed_42_d02/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\```json
{
  "val_loss": 0.004674740950576961,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\```

### 2026-04-10 14:36:19 EDT - doy_ablation seed=42 p=0.2 DOY_OFF pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_doy_ablation_5000/seed_42_d02_nodoy --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0006747921725036576,
  "train_seconds": 131.09032702445984
}
\`\`\`

### 2026-04-10 14:36:23 EDT - doy_ablation seed=42 p=0.2 DOY_OFF baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_42_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0006747924198862165,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:36:26 EDT - doy_ablation seed=42 p=0.2 DOY_OFF sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_42_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0036759445792995393,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:36:30 EDT - doy_ablation seed=42 p=0.2 DOY_OFF sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_42_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0007916061294963583,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:36:33 EDT - doy_ablation seed=42 p=0.2 DOY_OFF sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_42_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0009715607593534514,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:36:37 EDT - doy_ablation seed=42 p=0.2 DOY_OFF temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_42_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0025666203582659364,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-10 14:36:40 EDT - doy_ablation seed=42 p=0.2 DOY_OFF temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_42_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0040072109550237656,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-10 14:39:24 EDT - doy_ablation seed=7 p=0.2 DOY_OFF pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_doy_ablation_5000/seed_7_d02_nodoy --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0006555258441949263,
  "train_seconds": 126.85472106933594
}
\`\`\`

### 2026-04-10 14:39:27 EDT - doy_ablation seed=7 p=0.2 DOY_OFF baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_7_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0006555235886480659,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:39:30 EDT - doy_ablation seed=7 p=0.2 DOY_OFF sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_7_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.004438865813426673,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:39:33 EDT - doy_ablation seed=7 p=0.2 DOY_OFF sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_7_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0008103915170067921,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:39:37 EDT - doy_ablation seed=7 p=0.2 DOY_OFF sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_7_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0010981392406392843,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-10 14:39:40 EDT - doy_ablation seed=7 p=0.2 DOY_OFF temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_7_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.004383536288514733,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-10 14:39:43 EDT - doy_ablation seed=7 p=0.2 DOY_OFF temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_doy_ablation_5000/seed_7_d02_nodoy/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.006923273904249072,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-10 14:39:59 EDT - doy_ablation 2seed aggregate
- Command: aggregate exp_doy_ablation_5000_seed42.csv + exp_doy_ablation_5000_seed7.csv
- Output: artifacts/figures/exp_doy_ablation_5000_2seed_agg.csv

### 2026-04-11 13:20:33 EDT - s2_curriculum seed=42 use_doy=true pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_curriculum_5000/seed_42_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.5 --s2-blackout-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0010201257100561634,
  "train_seconds": 157.5156421661377
}
\`\`\`

### 2026-04-11 13:20:38 EDT - s2_curriculum seed=42 use_doy=true baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0010201260447502136,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:20:42 EDT - s2_curriculum seed=42 use_doy=true sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0003589004627428949,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:20:46 EDT - s2_curriculum seed=42 use_doy=true sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0012168054527137429,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:20:51 EDT - s2_curriculum seed=42 use_doy=true sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0015927393396850675,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:20:55 EDT - s2_curriculum seed=42 use_doy=true temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0024448485928587615,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:20:59 EDT - s2_curriculum seed=42 use_doy=true temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0035245760809630156,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:23:38 EDT - s2_curriculum seed=42 use_doy=false pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_curriculum_5000/seed_42_doy_off --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.5 --s2-blackout-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0009594391449354589,
  "train_seconds": 157.3699791431427
}
\`\`\`

### 2026-04-11 13:23:43 EDT - s2_curriculum seed=42 use_doy=false baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.000959439785219729,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:23:47 EDT - s2_curriculum seed=42 use_doy=false sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0007561384554719552,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:23:52 EDT - s2_curriculum seed=42 use_doy=false sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0010735454852692783,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:23:56 EDT - s2_curriculum seed=42 use_doy=false sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0014458756777457893,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:24:00 EDT - s2_curriculum seed=42 use_doy=false temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0017080589896067977,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:24:04 EDT - s2_curriculum seed=42 use_doy=false temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_42_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0021160825272090733,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:26:45 EDT - s2_curriculum seed=7 use_doy=true pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_curriculum_5000/seed_7_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.5 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 0.001315245492151007,
  "train_seconds": 158.53852009773254
}
\`\`\`

### 2026-04-11 13:26:50 EDT - s2_curriculum seed=7 use_doy=true baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0013152472965884954,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:26:54 EDT - s2_curriculum seed=7 use_doy=true sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0006725075945723802,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:26:58 EDT - s2_curriculum seed=7 use_doy=true sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0018900907307397574,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:27:02 EDT - s2_curriculum seed=7 use_doy=true sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0031201333040371537,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:27:07 EDT - s2_curriculum seed=7 use_doy=true temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0034702789853326976,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:27:11 EDT - s2_curriculum seed=7 use_doy=true temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.005078783608041704,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:29:55 EDT - s2_curriculum seed=7 use_doy=false pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_curriculum_5000/seed_7_doy_off --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.5 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0010199813987128437,
  "train_seconds": 162.15883708000183
}
\`\`\`

### 2026-04-11 13:30:00 EDT - s2_curriculum seed=7 use_doy=false baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0010199825192103162,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:30:04 EDT - s2_curriculum seed=7 use_doy=false sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0006957055011298507,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:30:08 EDT - s2_curriculum seed=7 use_doy=false sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0012188521795906126,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:30:12 EDT - s2_curriculum seed=7 use_doy=false sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00192155665718019,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:30:16 EDT - s2_curriculum seed=7 use_doy=false temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0020867622224614024,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:30:20 EDT - s2_curriculum seed=7 use_doy=false temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_curriculum_5000/seed_7_doy_off/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0027069858624599874,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:30:20 EDT - s2_curriculum aggregate
- Command: `aggregate exp_s2_curriculum_doy_5000_long.csv`
- Output:
\`\`\`json
{}
\`\`\`

### 2026-04-11 13:34:19 EDT - s2_blackout_sweep pmax=0.3 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout_sweep_5000/p03_seed_42_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0008359397907042876,
  "train_seconds": 149.3137228488922
}
\`\`\`

### 2026-04-11 13:34:23 EDT - s2_blackout_sweep pmax=0.3 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0008359397907042876,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:34:27 EDT - s2_blackout_sweep pmax=0.3 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0006940564198885113,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:34:31 EDT - s2_blackout_sweep pmax=0.3 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.001010265710647218,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:34:36 EDT - s2_blackout_sweep pmax=0.3 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0012539302406366915,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:34:40 EDT - s2_blackout_sweep pmax=0.3 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.002370812580920756,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:34:44 EDT - s2_blackout_sweep pmax=0.3 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0035980858956463635,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:37:27 EDT - s2_blackout_sweep pmax=0.3 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout_sweep_5000/p03_seed_7_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 0.000924095293157734,
  "train_seconds": 160.85237789154053
}
\`\`\`

### 2026-04-11 13:37:31 EDT - s2_blackout_sweep pmax=0.3 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0009240935323759913,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:37:35 EDT - s2_blackout_sweep pmax=0.3 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.001367663877317682,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:37:39 EDT - s2_blackout_sweep pmax=0.3 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0012992475822102278,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:37:43 EDT - s2_blackout_sweep pmax=0.3 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0020645306212827563,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:37:48 EDT - s2_blackout_sweep pmax=0.3 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.003321321797557175,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:37:52 EDT - s2_blackout_sweep pmax=0.3 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p03_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.005107555305585265,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:40:33 EDT - s2_blackout_sweep pmax=0.5 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.5 --s2-blackout-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0010201257100561634,
  "train_seconds": 159.17142605781555
}
\`\`\`

### 2026-04-11 13:40:37 EDT - s2_blackout_sweep pmax=0.5 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0010201260447502136,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:40:42 EDT - s2_blackout_sweep pmax=0.5 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0003589004627428949,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:40:46 EDT - s2_blackout_sweep pmax=0.5 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0012168054527137429,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:40:50 EDT - s2_blackout_sweep pmax=0.5 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0015927393396850675,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:40:54 EDT - s2_blackout_sweep pmax=0.5 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0024448485928587615,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:40:58 EDT - s2_blackout_sweep pmax=0.5 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0035245760809630156,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:43:35 EDT - s2_blackout_sweep pmax=0.5 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.5 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 0.001315245492151007,
  "train_seconds": 153.66320085525513
}
\`\`\`

### 2026-04-11 13:43:39 EDT - s2_blackout_sweep pmax=0.5 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0013152472965884954,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:43:43 EDT - s2_blackout_sweep pmax=0.5 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0006725075945723802,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:43:46 EDT - s2_blackout_sweep pmax=0.5 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0018900907307397574,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:43:50 EDT - s2_blackout_sweep pmax=0.5 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0031201333040371537,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:43:54 EDT - s2_blackout_sweep pmax=0.5 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0034702789853326976,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:43:57 EDT - s2_blackout_sweep pmax=0.5 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.005078783608041704,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:46:13 EDT - s2_blackout_sweep pmax=0.7 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout_sweep_5000/p07_seed_42_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.7 --s2-blackout-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0014598627749364823,
  "train_seconds": 133.9583718776703
}
\`\`\`

### 2026-04-11 13:46:17 EDT - s2_blackout_sweep pmax=0.7 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00145986262941733,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:46:20 EDT - s2_blackout_sweep pmax=0.7 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001894435154099483,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:46:23 EDT - s2_blackout_sweep pmax=0.7 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.001721282140351832,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:46:27 EDT - s2_blackout_sweep pmax=0.7 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.002382962207775563,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:46:30 EDT - s2_blackout_sweep pmax=0.7 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0027944542816840112,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:46:34 EDT - s2_blackout_sweep pmax=0.7 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00373337185010314,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:48:40 EDT - s2_blackout_sweep pmax=0.7 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout_sweep_5000/p07_seed_7_doy_on --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --use-doy --s2-blackout-max-p 0.7 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0021566966897808015,
  "train_seconds": 124.9954559803009
}
\`\`\`

### 2026-04-11 13:48:44 EDT - s2_blackout_sweep pmax=0.7 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0021566973300650716,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:48:47 EDT - s2_blackout_sweep pmax=0.7 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0003430664728512056,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:48:50 EDT - s2_blackout_sweep pmax=0.7 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0031935262959450483,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:48:54 EDT - s2_blackout_sweep pmax=0.7 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.005300373071804643,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 13:48:57 EDT - s2_blackout_sweep pmax=0.7 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.004083961364813149,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 13:49:01 EDT - s2_blackout_sweep pmax=0.7 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p07_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.005525792599655688,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 13:49:01 EDT - s2_blackout_sweep aggregate
- Command: `aggregate exp_s2_blackout_sweep_5000_long.csv`
- Output:
\`\`\`json
{}
\`\`\`

### 2026-04-11 14:37:27 EDT - s2_blackout03_doyoff seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout03_doyoff_5000/seed_42 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0007833211129764095,
  "train_seconds": 136.09263706207275
}
\`\`\`

### 2026-04-11 14:37:31 EDT - s2_blackout03_doyoff seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0007833207928342745,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:37:35 EDT - s2_blackout03_doyoff seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0014171493821777403,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:37:39 EDT - s2_blackout03_doyoff seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0008991945796879008,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:37:42 EDT - s2_blackout03_doyoff seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0011532876233104616,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:37:46 EDT - s2_blackout03_doyoff seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0016568009741604328,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 14:37:50 EDT - s2_blackout03_doyoff seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0022545066894963384,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 14:40:07 EDT - s2_blackout03_doyoff seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout03_doyoff_5000/seed_7 --epochs 5 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0008063057466642931,
  "train_seconds": 135.13181376457214
}
\`\`\`

### 2026-04-11 14:40:11 EDT - s2_blackout03_doyoff seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0008063058921834454,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:40:14 EDT - s2_blackout03_doyoff seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0014957326638977975,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:40:18 EDT - s2_blackout03_doyoff seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0009732291218824685,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:40:21 EDT - s2_blackout03_doyoff seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0014257599832490087,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 14:40:25 EDT - s2_blackout03_doyoff seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.002162650693207979,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 14:40:28 EDT - s2_blackout03_doyoff seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0030362310935743153,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 14:40:29 EDT - s2_blackout03_doyoff aggregate
- Command: `aggregate exp_s2_blackout03_doyoff_5000_long.csv`
- Output:
\`\`\`json
{}
\`\`\`

### 2026-04-11 15:10:21 EDT - s2_blackout03_doyoff_long20 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00013423158998193685,
  "train_seconds": 522.0556302070618
}
\`\`\`

### 2026-04-11 15:10:25 EDT - s2_blackout03_doyoff_long20 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00013423114432953298,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 15:10:28 EDT - s2_blackout03_doyoff_long20 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 1.4343756220114301e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 15:10:32 EDT - s2_blackout03_doyoff_long20 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001485560387664009,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 15:10:35 EDT - s2_blackout03_doyoff_long20 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00015869335402385332,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 15:10:39 EDT - s2_blackout03_doyoff_long20 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0011247261427342892,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 15:10:42 EDT - s2_blackout03_doyoff_long20 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0027715840260498226,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:08:59 EDT - s2_blackout03_doyoff_long20 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 9.441120528208558e-05,
  "train_seconds": 3495.165832042694
}
\`\`\`

### 2026-04-11 16:09:03 EDT - s2_blackout03_doyoff_long20 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.441231122764293e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:09:07 EDT - s2_blackout03_doyoff_long20 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 1.587643919265247e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:09:10 EDT - s2_blackout03_doyoff_long20 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010838064008567017,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:09:14 EDT - s2_blackout03_doyoff_long20 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001163207180070458,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:09:17 EDT - s2_blackout03_doyoff_long20 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0009923708130372688,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:09:21 EDT - s2_blackout03_doyoff_long20 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0024658364127390087,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:12:44 EDT - stress_eval doy_on_p05_e5 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.005024837213568389,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:12:48 EDT - stress_eval doy_on_p05_e5 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0021353805204853415,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:12:51 EDT - stress_eval doy_on_p05_e5 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_42_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.003553693532012403,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:12:55 EDT - stress_eval doy_off_p03_e5 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.0032661151490174234,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:12:58 EDT - stress_eval doy_off_p03_e5 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0025168127031065524,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:13:02 EDT - stress_eval doy_off_p03_e5 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.003447773866355419,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:13:06 EDT - stress_eval doy_off_p03_e20 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.006579324952326715,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:13:09 EDT - stress_eval doy_off_p03_e20 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0011609010689426214,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:13:13 EDT - stress_eval doy_off_p03_e20 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0030798770603723824,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:13:16 EDT - stress_eval doy_on_p05_e5 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.006986370659433305,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:13:20 EDT - stress_eval doy_on_p05_e5 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.002598095335997641,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:13:24 EDT - stress_eval doy_on_p05_e5 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout_sweep_5000/p05_seed_7_doy_on/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0041221423307433724,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:13:27 EDT - stress_eval doy_off_p03_e5 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.004313053213991225,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:13:31 EDT - stress_eval doy_off_p03_e5 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0028451745165511966,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:13:34 EDT - stress_eval doy_off_p03_e5 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0038579345564357936,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:13:38 EDT - stress_eval doy_off_p03_e20 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00491139548830688,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:13:42 EDT - stress_eval doy_off_p03_e20 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0007167167059378698,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:13:45 EDT - stress_eval doy_off_p03_e20 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0019292934739496559,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:13:45 EDT - stress_eval aggregate
- Command: `aggregate stress eval results`
- Output:
\`\`\`json
{}
\`\`\`

### 2026-04-11 16:15:30 EDT - s2_blackout03_doyoff_long20 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_s2_blackout03_doyoff_5000_long20/seed_7 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 9.441120528208558e-05,
  "train_seconds": 503.1204171180725
}
\`\`\`

### 2026-04-11 16:49:30 EDT - doy_off_p03_e20_tdropaug05 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.5 --temporal-drop-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00015599057587678544,
  "train_seconds": 499.95320773124695
}
\`\`\`

### 2026-04-11 16:49:33 EDT - doy_off_p03_e20_tdropaug05 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00015598997924826108,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:49:37 EDT - doy_off_p03_e20_tdropaug05 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 6.733371810696553e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:49:40 EDT - doy_off_p03_e20_tdropaug05 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00017521915651741438,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:49:43 EDT - doy_off_p03_e20_tdropaug05 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00019036498269997537,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:49:46 EDT - doy_off_p03_e20_tdropaug05 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00016742524894652888,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:49:50 EDT - doy_off_p03_e20_tdropaug05 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00017320805636700243,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:49:53 EDT - doy_off_p03_e20_tdropaug05 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00022406733842217363,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:49:56 EDT - doy_off_p03_e20_tdropaug05 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.590813791262917e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:49:59 EDT - doy_off_p03_e20_tdropaug05 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.860850695986301e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:58:30 EDT - doy_off_p03_e20_tdropaug05 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.5 --temporal-drop-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 8.634962432552129e-05,
  "train_seconds": 508.7300329208374
}
\`\`\`

### 2026-04-11 16:58:33 EDT - doy_off_p03_e20_tdropaug05 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 8.634733603685163e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:58:37 EDT - doy_off_p03_e20_tdropaug05 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 5.777593560196692e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:58:40 EDT - doy_off_p03_e20_tdropaug05 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.92366076388862e-05,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:58:44 EDT - doy_off_p03_e20_tdropaug05 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010806420505105052,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 16:58:47 EDT - doy_off_p03_e20_tdropaug05 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00011220262786082458,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:58:50 EDT - doy_off_p03_e20_tdropaug05 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00015667422849219292,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 16:58:54 EDT - doy_off_p03_e20_tdropaug05 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.0002890135001507588,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 16:58:58 EDT - doy_off_p03_e20_tdropaug05 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.978441524552181e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 16:59:01 EDT - doy_off_p03_e20_tdropaug05 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00013127416059433017,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 17:07:26 EDT - doy_off_p03_e20_tdropaug07 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.7 --temporal-drop-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00017766947712516412,
  "train_seconds": 502.3045492172241
}
\`\`\`

### 2026-04-11 17:07:29 EDT - doy_off_p03_e20_tdropaug07 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001776677709131036,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:07:33 EDT - doy_off_p03_e20_tdropaug07 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 7.84627663961146e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:07:36 EDT - doy_off_p03_e20_tdropaug07 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001998821499000769,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:07:40 EDT - doy_off_p03_e20_tdropaug07 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00022319001800497063,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:07:43 EDT - doy_off_p03_e20_tdropaug07 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00017840732834883966,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 17:07:46 EDT - doy_off_p03_e20_tdropaug07 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00016086251707747579,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 17:07:49 EDT - doy_off_p03_e20_tdropaug07 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00015700502990512177,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 17:07:53 EDT - doy_off_p03_e20_tdropaug07 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 7.063418343022931e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 17:07:56 EDT - doy_off_p03_e20_tdropaug07 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.185525439330377e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 17:16:13 EDT - doy_off_p03_e20_tdropaug07 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.7 --temporal-drop-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 8.963159598351922e-05,
  "train_seconds": 495.8905100822449
}
\`\`\`

### 2026-04-11 17:16:17 EDT - doy_off_p03_e20_tdropaug07 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 8.963197069533635e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:16:20 EDT - doy_off_p03_e20_tdropaug07 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 6.503597796836402e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:16:23 EDT - doy_off_p03_e20_tdropaug07 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010252950960421003,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:16:26 EDT - doy_off_p03_e20_tdropaug07 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00011854691183543764,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 17:16:30 EDT - doy_off_p03_e20_tdropaug07 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00010216335431323387,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 17:16:33 EDT - doy_off_p03_e20_tdropaug07 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00011124942466267385,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 17:16:36 EDT - doy_off_p03_e20_tdropaug07 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00015119995441637002,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 17:16:39 EDT - doy_off_p03_e20_tdropaug07 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 5.7062629821302835e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 17:16:43 EDT - doy_off_p03_e20_tdropaug07 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.109592818072997e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 17:16:43 EDT - tdrop_train_aug aggregate
- Command: `aggregate + extended model selection tables`
- Output:
\`\`\`json
{}
\`\`\`

### 2026-04-11 21:23:29 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 5 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00016464307554997504,
  "train_seconds": 506.87439799308777
}
\`\`\`

### 2026-04-11 21:23:33 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001646436248847749,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:23:36 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 7.13882745912997e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:23:39 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00018550284221419133,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:23:43 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0002034920507867355,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:23:46 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00017131144340964966,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:23:49 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00016305835742969066,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:23:53 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00017925166321219876,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 21:23:57 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.66406995151192e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:24:01 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.536142922821455e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:32:22 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 10 --seed 42`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0001632128878554795,
  "train_seconds": 499.22724294662476
}
\`\`\`

### 2026-04-11 21:32:25 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00016321168004651554,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:32:28 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 7.889316839282401e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:32:32 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00018218376499135047,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:32:36 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00020014113033539616,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:32:40 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00017505599680589512,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:32:43 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00017533101345179603,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:32:47 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00021911555450060405,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 21:32:50 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.978453893680125e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:32:53 EDT - doy_off_p03_e20_tdropaug06_w10 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.753739686857443e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:41:44 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 5 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 8.760652781347744e-05,
  "train_seconds": 528.6703269481659
}
\`\`\`

### 2026-04-11 21:41:47 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 8.760673335928004e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:41:51 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 6.311462311714422e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:41:54 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010027698044723365,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:41:57 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001122895901062293,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:42:01 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00010462560203450266,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:42:04 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00012612992395588662,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:42:08 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00020102404232602566,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 21:42:11 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 5.954287917120382e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:42:14 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.668948223406915e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:50:56 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 10 --seed 7`
- Output:
\`\`\`json
{
  "best_val_loss": 9.709787991596386e-05,
  "train_seconds": 519.4971101284027
}
\`\`\`

### 2026-04-11 21:50:59 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.70982091530459e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:51:03 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 6.268783909035847e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:51:07 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00011115673260064796,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:51:11 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00012195692761451937,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-11 21:51:14 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00011713557796610985,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:51:18 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00014780726633034647,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:51:21 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00024809256865410134,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-11 21:51:25 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.0324729020067025e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-11 21:51:29 EDT - doy_off_p03_e20_tdropaug06_w10 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w10_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00010581574497336987,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-11 21:52:16 EDT - tdrop06_warmup_sweep aggregate (recovery rerun)
- Command: `recompute aggregate/main/stress tables from exp_tdrop06_warmup_sweep_5000_long.csv`
- Output:
\`\`\`json
{"status":"ok","note":"Recovered from prior KeyError(run_name) in initial aggregation script."}
\`\`\`

### 2026-04-12 02:42:26 EDT - doy_off_p03_e20_tdropaug05 seed=42 pretrain (reused)
- Command: `reuse /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt`
- Output:
\`\`\`json
{"best_val_loss": 0.00015599057587678544, "train_seconds": 499.95320773124695}
\`\`\`

### 2026-04-12 02:42:30 EDT - doy_off_p03_e20_tdropaug05 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00015598997924826108,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:42:33 EDT - doy_off_p03_e20_tdropaug05 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 6.733371810696553e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:42:36 EDT - doy_off_p03_e20_tdropaug05 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00017521915651741438,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:42:40 EDT - doy_off_p03_e20_tdropaug05 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00019036498269997537,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:42:43 EDT - doy_off_p03_e20_tdropaug05 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00016742524894652888,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:42:46 EDT - doy_off_p03_e20_tdropaug05 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00017320805636700243,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:42:50 EDT - doy_off_p03_e20_tdropaug05 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00022406733842217363,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 02:42:54 EDT - doy_off_p03_e20_tdropaug05 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.590813791262917e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:42:58 EDT - doy_off_p03_e20_tdropaug05 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.860850695986301e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:42:58 EDT - doy_off_p03_e20_tdropaug07 seed=42 pretrain (reused)
- Command: `reuse /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt`
- Output:
\`\`\`json
{"best_val_loss": 0.00017766947712516412, "train_seconds": 502.3045492172241}
\`\`\`

### 2026-04-12 02:43:01 EDT - doy_off_p03_e20_tdropaug07 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001776677709131036,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:04 EDT - doy_off_p03_e20_tdropaug07 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 7.84627663961146e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:07 EDT - doy_off_p03_e20_tdropaug07 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001998821499000769,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:11 EDT - doy_off_p03_e20_tdropaug07 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00022319001800497063,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:15 EDT - doy_off_p03_e20_tdropaug07 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00017840732834883966,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:43:18 EDT - doy_off_p03_e20_tdropaug07 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00016086251707747579,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:43:22 EDT - doy_off_p03_e20_tdropaug07 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00015700502990512177,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 02:43:26 EDT - doy_off_p03_e20_tdropaug07 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 7.063418343022931e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:43:31 EDT - doy_off_p03_e20_tdropaug07 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.185525439330377e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:43:31 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 pretrain (reused)
- Command: `reuse /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt`
- Output:
\`\`\`json
{"best_val_loss": 0.00016464307554997504, "train_seconds": 506.87439799308777}
\`\`\`

### 2026-04-12 02:43:35 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001646436248847749,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:38 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 7.13882745912997e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:42 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00018550284221419133,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:46 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0002034920507867355,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:43:49 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00017131144340964966,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:43:53 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00016305835742969066,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:43:57 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00017925166321219876,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 02:44:01 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.66406995151192e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:44:05 EDT - doy_off_p03_e20_tdropaug06_w5 seed=42 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_42/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 42 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.536142922821455e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:44:06 EDT - doy_off_p03_e20_tdropaug05 seed=7 pretrain (reused)
- Command: `reuse /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt`
- Output:
\`\`\`json
{"best_val_loss": 8.634962432552129e-05, "train_seconds": 508.7300329208374}
\`\`\`

### 2026-04-12 02:44:09 EDT - doy_off_p03_e20_tdropaug05 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 8.634733603685163e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:44:13 EDT - doy_off_p03_e20_tdropaug05 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 5.777593560196692e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:44:17 EDT - doy_off_p03_e20_tdropaug05 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.92366076388862e-05,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:44:21 EDT - doy_off_p03_e20_tdropaug05 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010806420505105052,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:44:25 EDT - doy_off_p03_e20_tdropaug05 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00011220262786082458,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:44:28 EDT - doy_off_p03_e20_tdropaug05 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00015667422849219292,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:44:33 EDT - doy_off_p03_e20_tdropaug05 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.0002890135001507588,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 02:44:36 EDT - doy_off_p03_e20_tdropaug05 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.978441524552181e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:44:40 EDT - doy_off_p03_e20_tdropaug05 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00013127416059433017,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:44:40 EDT - doy_off_p03_e20_tdropaug07 seed=7 pretrain (reused)
- Command: `reuse /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt`
- Output:
\`\`\`json
{"best_val_loss": 8.963159598351922e-05, "train_seconds": 495.8905100822449}
\`\`\`

### 2026-04-12 02:44:44 EDT - doy_off_p03_e20_tdropaug07 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 8.963197069533635e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:44:48 EDT - doy_off_p03_e20_tdropaug07 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 6.503597796836402e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:44:52 EDT - doy_off_p03_e20_tdropaug07 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010252950960421003,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:44:56 EDT - doy_off_p03_e20_tdropaug07 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00011854691183543764,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:45:00 EDT - doy_off_p03_e20_tdropaug07 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00010216335431323387,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:45:05 EDT - doy_off_p03_e20_tdropaug07 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00011124942466267385,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:45:09 EDT - doy_off_p03_e20_tdropaug07 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00015119995441637002,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 02:45:13 EDT - doy_off_p03_e20_tdropaug07 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 5.7062629821302835e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:45:17 EDT - doy_off_p03_e20_tdropaug07 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.109592818072997e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:45:17 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 pretrain (reused)
- Command: `reuse /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt`
- Output:
\`\`\`json
{"best_val_loss": 8.760652781347744e-05, "train_seconds": 528.6703269481659}
\`\`\`

### 2026-04-12 02:45:21 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 8.760673335928004e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:45:25 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 6.311462311714422e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:45:29 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010027698044723365,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:45:33 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001122895901062293,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:45:37 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00010462560203450266,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:45:41 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00012612992395588662,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:45:45 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00020102404232602566,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 02:45:49 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 5.954287917120382e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:45:53 EDT - doy_off_p03_e20_tdropaug06_w5 seed=7 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_7/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 7 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.668948223406915e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:55:06 EDT - doy_off_p03_e20_tdropaug05 seed=11 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.5 --temporal-drop-warmup-epochs 5 --seed 11`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0001195554286823608,
  "train_seconds": 551.1311039924622
}
\`\`\`

### 2026-04-12 02:55:09 EDT - doy_off_p03_e20_tdropaug05 seed=11 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00011955564514209982,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:55:13 EDT - doy_off_p03_e20_tdropaug05 seed=11 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 4.973998147761449e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:55:17 EDT - doy_off_p03_e20_tdropaug05 seed=11 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001358870831609238,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:55:20 EDT - doy_off_p03_e20_tdropaug05 seed=11 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00014473331248154864,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 02:55:24 EDT - doy_off_p03_e20_tdropaug05 seed=11 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00014060750982025638,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:55:28 EDT - doy_off_p03_e20_tdropaug05 seed=11 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00017037035286193714,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 02:55:31 EDT - doy_off_p03_e20_tdropaug05 seed=11 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00027738635253626853,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 02:55:35 EDT - doy_off_p03_e20_tdropaug05 seed=11 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 7.511574767704587e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 02:55:38 EDT - doy_off_p03_e20_tdropaug05 seed=11 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.000126368553537759,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:04:12 EDT - doy_off_p03_e20_tdropaug07 seed=11 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.7 --temporal-drop-warmup-epochs 5 --seed 11`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00012914976286992896,
  "train_seconds": 511.5679841041565
}
\`\`\`

### 2026-04-12 03:04:15 EDT - doy_off_p03_e20_tdropaug07 seed=11 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00012915005754621234,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:04:18 EDT - doy_off_p03_e20_tdropaug07 seed=11 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 5.454310758068459e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:04:21 EDT - doy_off_p03_e20_tdropaug07 seed=11 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001454122866562102,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:04:25 EDT - doy_off_p03_e20_tdropaug07 seed=11 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00015876679390203208,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:04:28 EDT - doy_off_p03_e20_tdropaug07 seed=11 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00014115856538410299,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:04:31 EDT - doy_off_p03_e20_tdropaug07 seed=11 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00014873294639983214,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:04:34 EDT - doy_off_p03_e20_tdropaug07 seed=11 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00019301164138596505,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 03:04:38 EDT - doy_off_p03_e20_tdropaug07 seed=11 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.509046215796843e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:04:41 EDT - doy_off_p03_e20_tdropaug07 seed=11 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.522024083707947e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:13:30 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 5 --seed 11`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00012310036072449293,
  "train_seconds": 527.4919390678406
}
\`\`\`

### 2026-04-12 03:13:34 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00012310184138186742,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:13:38 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 5.637569211103255e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:13:41 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00014046967044123448,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:13:45 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00015209688353934325,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:13:48 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00013696825408260338,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:13:52 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0001512686358182691,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:13:55 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.0002146472252206877,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 03:13:59 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 7.175046266638674e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:14:02 EDT - doy_off_p03_e20_tdropaug06_w5 seed=11 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_11/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 11 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0001068950914486777,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:22:26 EDT - doy_off_p03_e20_tdropaug05 seed=19 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.5 --temporal-drop-warmup-epochs 5 --seed 19`
- Output:
\`\`\`json
{
  "best_val_loss": 9.064339428732637e-05,
  "train_seconds": 502.00983786582947
}
\`\`\`

### 2026-04-12 03:22:30 EDT - doy_off_p03_e20_tdropaug05 seed=19 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.064312507689465e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:22:33 EDT - doy_off_p03_e20_tdropaug05 seed=19 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.038060670718551e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:22:37 EDT - doy_off_p03_e20_tdropaug05 seed=19 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010488039151823614,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:22:40 EDT - doy_off_p03_e20_tdropaug05 seed=19 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00013951250002719462,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:22:44 EDT - doy_off_p03_e20_tdropaug05 seed=19 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 9.559376849210821e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:22:48 EDT - doy_off_p03_e20_tdropaug05 seed=19 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00010614982238621451,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:22:52 EDT - doy_off_p03_e20_tdropaug05 seed=19 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00017367190594086424,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 03:22:56 EDT - doy_off_p03_e20_tdropaug05 seed=19 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 7.280997851921711e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:22:59 EDT - doy_off_p03_e20_tdropaug05 seed=19 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00010076043690787628,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:31:20 EDT - doy_off_p03_e20_tdropaug07 seed=19 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.7 --temporal-drop-warmup-epochs 5 --seed 19`
- Output:
\`\`\`json
{
  "best_val_loss": 9.882225640467368e-05,
  "train_seconds": 499.27754759788513
}
\`\`\`

### 2026-04-12 03:31:24 EDT - doy_off_p03_e20_tdropaug07 seed=19 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.882062113319989e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:31:27 EDT - doy_off_p03_e20_tdropaug07 seed=19 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00013128334830980748,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:31:31 EDT - doy_off_p03_e20_tdropaug07 seed=19 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00011509772230056114,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:31:34 EDT - doy_off_p03_e20_tdropaug07 seed=19 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00015700600124546327,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:31:38 EDT - doy_off_p03_e20_tdropaug07 seed=19 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 9.993081584980246e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:31:41 EDT - doy_off_p03_e20_tdropaug07 seed=19 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.190375931211747e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:31:45 EDT - doy_off_p03_e20_tdropaug07 seed=19 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00010415291035315022,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 03:31:48 EDT - doy_off_p03_e20_tdropaug07 seed=19 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 8.600047112850007e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:31:51 EDT - doy_off_p03_e20_tdropaug07 seed=19 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.600194814789575e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:40:04 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 5 --seed 19`
- Output:
\`\`\`json
{
  "best_val_loss": 9.500293708697427e-05,
  "train_seconds": 490.8006057739258
}
\`\`\`

### 2026-04-12 03:40:07 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.500410851615015e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:40:10 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010745783220045269,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:40:14 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00011020054989785422,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:40:17 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00015074228576850146,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:40:20 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 9.54001116042491e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:40:23 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.31167887756601e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:40:27 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00012492073437897488,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 03:40:30 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 7.517005178669933e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:40:33 EDT - doy_off_p03_e20_tdropaug06_w5 seed=19 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_19/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 19 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 8.743235048314091e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:49:23 EDT - doy_off_p03_e20_tdropaug05 seed=23 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.5 --temporal-drop-warmup-epochs 5 --seed 23`
- Output:
\`\`\`json
{
  "best_val_loss": 8.76601152413059e-05,
  "train_seconds": 527.9554979801178
}
\`\`\`

### 2026-04-12 03:49:26 EDT - doy_off_p03_e20_tdropaug05 seed=23 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 8.766043538344093e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:49:29 EDT - doy_off_p03_e20_tdropaug05 seed=23 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 7.367663238255773e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:49:33 EDT - doy_off_p03_e20_tdropaug05 seed=23 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010316802217857912,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:49:36 EDT - doy_off_p03_e20_tdropaug05 seed=23 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00013010315160499886,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:49:39 EDT - doy_off_p03_e20_tdropaug05 seed=23 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00010152669347007759,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:49:43 EDT - doy_off_p03_e20_tdropaug05 seed=23 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0001119681510317605,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:49:46 EDT - doy_off_p03_e20_tdropaug05 seed=23 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.0001646406926738564,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 03:49:49 EDT - doy_off_p03_e20_tdropaug05 seed=23 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 7.33843189664185e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:49:53 EDT - doy_off_p03_e20_tdropaug05 seed=23 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 9.350584878120571e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:58:13 EDT - doy_off_p03_e20_tdropaug07 seed=23 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.7 --temporal-drop-warmup-epochs 5 --seed 23`
- Output:
\`\`\`json
{
  "best_val_loss": 9.960335955838673e-05,
  "train_seconds": 498.960364818573
}
\`\`\`

### 2026-04-12 03:58:17 EDT - doy_off_p03_e20_tdropaug07 seed=23 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.960238821804523e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:58:20 EDT - doy_off_p03_e20_tdropaug07 seed=23 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.408677760802675e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:58:23 EDT - doy_off_p03_e20_tdropaug07 seed=23 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00011662108954624273,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:58:26 EDT - doy_off_p03_e20_tdropaug07 seed=23 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00014731812188983895,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 03:58:30 EDT - doy_off_p03_e20_tdropaug07 seed=23 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00010884388393606059,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:58:33 EDT - doy_off_p03_e20_tdropaug07 seed=23 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00010072933582705446,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 03:58:36 EDT - doy_off_p03_e20_tdropaug07 seed=23 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 9.784747635421809e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 03:58:39 EDT - doy_off_p03_e20_tdropaug07 seed=23 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 8.278723180410452e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 03:58:43 EDT - doy_off_p03_e20_tdropaug07 seed=23 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 7.866013038437814e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 04:06:59 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 5 --seed 23`
- Output:
\`\`\`json
{
  "best_val_loss": 9.38574448809959e-05,
  "train_seconds": 494.80290699005127
}
\`\`\`

### 2026-04-12 04:07:03 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 9.385761040903162e-05,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:07:06 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 7.139055742300116e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:07:09 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010854316315089818,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:07:12 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00013537133781937882,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:07:15 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0001026607696985593,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 04:07:18 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.0001002935223368695,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 04:07:22 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00011517084203660488,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 04:07:25 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 6.80661978549324e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 04:07:28 EDT - doy_off_p03_e20_tdropaug06_w5 seed=23 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_23/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 23 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 7.513609125453513e-05,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 04:16:05 EDT - doy_off_p03_e20_tdropaug05 seed=101 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.5 --temporal-drop-warmup-epochs 5 --seed 101`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00010055462917080149,
  "train_seconds": 514.680902004242
}
\`\`\`

### 2026-04-12 04:16:08 EDT - doy_off_p03_e20_tdropaug05 seed=101 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010055537313746754,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:16:11 EDT - doy_off_p03_e20_tdropaug05 seed=101 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00012074235019099433,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:16:14 EDT - doy_off_p03_e20_tdropaug05 seed=101 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001272675726795569,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:16:18 EDT - doy_off_p03_e20_tdropaug05 seed=101 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00014716905207023956,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 04:16:21 EDT - doy_off_p03_e20_tdropaug05 seed=101 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00012368378884275444,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 04:16:24 EDT - doy_off_p03_e20_tdropaug05 seed=101 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00013963465607957914,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 04:16:27 EDT - doy_off_p03_e20_tdropaug05 seed=101 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.0002550081044319086,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 04:16:31 EDT - doy_off_p03_e20_tdropaug05 seed=101 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00012550386963994242,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 04:16:34 EDT - doy_off_p03_e20_tdropaug05 seed=101 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop05_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00017096747615141794,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 17:10:43 EDT - doy_off_p03_e20_tdropaug07 seed=101 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.7 --temporal-drop-warmup-epochs 5 --seed 101`
- Output:
\`\`\`json
{
  "best_val_loss": 0.00010315814506611787,
  "train_seconds": 519.2130599021912
}
\`\`\`

### 2026-04-12 17:10:47 EDT - doy_off_p03_e20_tdropaug07 seed=101 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010315855615772307,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:10:50 EDT - doy_off_p03_e20_tdropaug07 seed=101 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00014487101725535467,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:10:53 EDT - doy_off_p03_e20_tdropaug07 seed=101 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00013005753135075793,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:10:57 EDT - doy_off_p03_e20_tdropaug07 seed=101 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001532737514935434,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:11:00 EDT - doy_off_p03_e20_tdropaug07 seed=101 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0001235645486303838,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 17:11:04 EDT - doy_off_p03_e20_tdropaug07 seed=101 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00011624016588029917,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 17:11:07 EDT - doy_off_p03_e20_tdropaug07 seed=101 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00013603927072836086,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 17:11:11 EDT - doy_off_p03_e20_tdropaug07 seed=101 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00011808106864918955,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 17:11:15 EDT - doy_off_p03_e20_tdropaug07 seed=101 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop_train_aug_5000/tdrop07_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00012165173575340305,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 17:20:16 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 pretrain
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli pretrain-next --zarr-path /tmp/cropharvest_5000.zarr --output-dir /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101 --epochs 20 --batch-size 128 --num-workers 0 --device cpu --model-dim 128 --encoder-hidden 64 --num-layers 2 --num-heads 8 --modality-dropout-p 0.2 --disable-doy --s2-blackout-max-p 0.3 --s2-blackout-warmup-epochs 5 --temporal-drop-max-fraction 0.6 --temporal-drop-warmup-epochs 5 --seed 101`
- Output:
\`\`\`json
{
  "best_val_loss": 0.0001002269345917739,
  "train_seconds": 538.7006731033325
}
\`\`\`

### 2026-04-12 17:20:20 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 baseline_none_d0
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00010022629066952504,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:20:24 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 sensor_off_s2
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.00013605020649265498,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:20:28 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 sensor_off_s1
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s1 --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001263538379134843,
  "num_val_batches": 4,
  "sensor_off": "s1",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:20:32 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 sensor_off_climate
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off climate --temporal-drop-fraction 0.0`
- Output:
\`\`\`json
{
  "val_loss": 0.0001471365940233227,
  "num_val_batches": 4,
  "sensor_off": "climate",
  "temporal_drop_fraction": 0.0
}
\`\`\`

### 2026-04-12 17:20:36 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 temporal_drop_30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.0001227667453349568,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 17:20:39 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 temporal_drop_50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00012392175631248392,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 17:20:43 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 temporal_drop_70
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off none --temporal-drop-fraction 0.7`
- Output:
\`\`\`json
{
  "val_loss": 0.00017815294631873257,
  "num_val_batches": 4,
  "sensor_off": "none",
  "temporal_drop_fraction": 0.7
}
\`\`\`

### 2026-04-12 17:20:46 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 s2_off_tdrop30
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.3`
- Output:
\`\`\`json
{
  "val_loss": 0.00011905093742825557,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.3
}
\`\`\`

### 2026-04-12 17:20:50 EDT - doy_off_p03_e20_tdropaug06_w5 seed=101 s2_off_tdrop50
- Command: `PYTHONPATH=src .venv/bin/python -m surf.cli eval-next --zarr-path /tmp/cropharvest_5000.zarr --checkpoint-path /tmp/exp_tdrop06_warmup_sweep_5000/w5_seed_101/best_checkpoint.pt --device cpu --batch-size 128 --num-workers 0 --seed 101 --sensor-off s2 --temporal-drop-fraction 0.5`
- Output:
\`\`\`json
{
  "val_loss": 0.00013918180047767237,
  "num_val_batches": 4,
  "sensor_off": "s2",
  "temporal_drop_fraction": 0.5
}
\`\`\`

### 2026-04-12 17:20:50 EDT - seedexp_tdrop_models aggregate (resume completion)
- Command: `aggregate + selection tables`
- Output:
\`\`\`json
{}
\`\`\`

### 2026-04-12 17:28:00 EDT - download ssl4eo_s12_v1_1_train_mini
- Command: `huggingface-cli download embed2scale/SSL4EO-S12-v1.1 --repo-type dataset --local-dir data/raw/benchmarks/ssl4eo_s12_v1_1_train_mini --include train/S1GRD/ssl4eos12_shard_000001.tar train/S1GRD/ssl4eos12_shard_000002.tar train/S1GRD/ssl4eos12_shard_000003.tar train/S2L2A/ssl4eos12_shard_000001.tar train/S2L2A/ssl4eos12_shard_000002.tar train/S2L2A/ssl4eos12_shard_000003.tar train/NDVI/ssl4eos12_shard_000001.tar train/NDVI/ssl4eos12_shard_000002.tar train/NDVI/ssl4eos12_shard_000003.tar train_metadata.parquet README.md`
- Output:
\`\`\`json
{
  "status": "ok",
  "local_dir": "data/raw/benchmarks/ssl4eo_s12_v1_1_train_mini",
  "files_fetched": 11
}
\`\`\`

### 2026-04-12 17:30:00 EDT - download bigearthnet_mini
- Command: `huggingface-cli download lc-col/bigearthnet --repo-type dataset --local-dir data/raw/benchmarks/bigearthnet_mini --include README.md bigearthnet_hdf5_val.csv bigearthnet_val_p0.hdf5.gz`
- Output:
\`\`\`json
{
  "status": "ok",
  "local_dir": "data/raw/benchmarks/bigearthnet_mini",
  "files_fetched": 3
}
\`\`\`

### 2026-04-12 17:31:00 EDT - download eurosat_torchvision
- Command: `python -c 'from torchvision.datasets import EuroSAT; EuroSAT(\"data/raw/benchmarks/eurosat\", download=True)'`
- Output:
\`\`\`json
{
  "status": "ok",
  "samples": 27000,
  "classes": 10,
  "local_dir": "data/raw/benchmarks/eurosat"
}
\`\`\`

### 2026-04-12 17:34:00 EDT - build_cropharvest_full_zarr_complete
- Command: `.venv/bin/python -m surf.cli build-cropharvest-zarr --arrays-dir data/raw/cropharvest/features/arrays --output-zarr data/processed/cropharvest_multimodal_full.zarr --max-samples -1`
- Output:
\`\`\`json
{
  "status": "ok",
  "output_zarr": "data/processed/cropharvest_multimodal_full.zarr",
  "num_patches": 67692,
  "num_skipped": 1
}
\`\`\`

### 2026-04-12 17:36:00 EDT - refresh_local_dataset_inventory
- Command: `python inventory refresh script -> artifacts/figures/local_dataset_inventory_2026-04-12.csv`
- Output:
\`\`\`json
{
  "status": "ok",
  "inventory_csv": "artifacts/figures/local_dataset_inventory_2026-04-12.csv"
}
\`\`\`

### 2026-04-13 19:22:00 EDT - literature_results_reference_refresh
- Command: `lookup recent EO/agri FM papers (arXiv/OpenReview where available), extract reported headline results, and compile beginner-friendly reference markdown`
- Output:
\`\`\`json
{
  "status": "ok",
  "output_file": "docs/current_papers_results_reference_2026-04-13.md",
  "notes": "Includes competitor papers + reported metrics/claims + recommended next experiment."
}
\`\`\`

### 2026-04-13 19:33:00 EDT - launched_fewshot_probe_pipeline_background
- Command: `nohup env PYTHONPATH=src .venv/bin/python tools/run_fewshot_probe_pipeline.py > artifacts/logs/fewshot_probe_pipeline.log 2>&1 &`
- Output:
\`\`\`json
{
  "status": "started",
  "pid_file": "artifacts/logs/fewshot_probe_pipeline.pid",
  "log_file": "artifacts/logs/fewshot_probe_pipeline.log"
}
\`\`\`
