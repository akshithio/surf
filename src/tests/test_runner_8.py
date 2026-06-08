import json
import multiprocessing as mp
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest
import torch


def _runner():
    path = Path(__file__).resolve().parents[2] / "runners/[8].py"
    spec = spec_from_file_location("runner_8_tests", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _test_worker_ok(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    result_path = Path(str(preprocess_qa["output_dir"])) / f"worker_{gpu_id}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("[]")
    q.put({"gpu_id": gpu_id, "success": True, "result_path": str(result_path)})


def _test_worker_fail_immediate(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    q.put({"gpu_id": gpu_id, "success": False, "error": "immediate failure"})


def _test_worker_large_payload(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    result_path = Path(str(preprocess_qa["output_dir"])) / f"worker_{gpu_id}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"arm": a, "big": list(range(20000))} for a in worker_arms for _ in range(50)]
    result_path.write_text(json.dumps(data))
    q.put({"gpu_id": gpu_id, "success": True, "result_path": str(result_path)})


def _test_worker_sleep_then_fail(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    import time
    if gpu_id == 0:
        result_path = Path(str(preprocess_qa["output_dir"])) / f"worker_{gpu_id}_results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("[]")
        q.put({"gpu_id": gpu_id, "success": True, "result_path": str(result_path)})
    else:
        time.sleep(2)
        q.put({"gpu_id": gpu_id, "success": False, "error": "GPU 1 crashed"})


def _test_worker_os_exit_before_status(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    import os
    os._exit(7)


def _test_worker_success_then_os_exit(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    import os, time
    result_path = Path(str(preprocess_qa["output_dir"])) / f"worker_{gpu_id}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("[]")
    q.put({"gpu_id": gpu_id, "success": True, "result_path": str(result_path)})
    time.sleep(1)
    os._exit(7)


def _test_worker_success_then_hang(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    import time
    result_path = Path(str(preprocess_qa["output_dir"])) / f"worker_{gpu_id}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("[]")
    q.put({"gpu_id": gpu_id, "success": True, "result_path": str(result_path)})
    while True:
        time.sleep(3600)


def _test_worker_gpu0_hangs_gpu1_fails(gpu_id: int, worker_arms: list[str], q: mp.Queue, preprocess_qa: dict) -> None:
    if gpu_id == 0:
        _test_worker_success_then_hang(gpu_id, worker_arms, q, preprocess_qa)
    else:
        q.put({"gpu_id": gpu_id, "success": False, "error": "GPU 1 crashed"})



def _stats(scale: float = 1.0) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "s2": {
            "B2": {"min": 0.0, "max": scale, "std": scale / 4, "p10": scale / 10, "p90": scale * 0.9},
            "NDVI": {"min": -0.8, "max": 0.9, "std": 0.3, "p10": -0.4, "p90": 0.7},
        },
        "s1": {
            "VV": {"min": -30.0 * scale, "max": -1.0 * scale, "std": 5.0 * scale, "p10": -20.0 * scale, "p90": -5.0 * scale},
        },
    }


def test_target_mask_fraction_matches_across_enabled_arms() -> None:
    runner = _runner()
    device = torch.device("cpu")
    fractions = []
    for name in runner.DEFAULT_ARMS:
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        mask = runner._target_mask(runner.ARM_SPECS[name], 64, 4, device, generator)
        fractions.append(float(mask.float().mean()))
    assert fractions == [runner.TARGET_MASK_FRACTION] * len(fractions)


def test_d_arm_spatiotemporal_blocks_are_contiguous_and_exact_budget() -> None:
    runner = _runner()
    device = torch.device("cpu")
    spec = runner.ARM_SPECS["D_spatiotemporal"]
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    mask_flat = runner._target_mask(spec, 64, 4, device, generator)
    mask = mask_flat.reshape(64, 4, 4, 4)
    quadrants = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for row in range(64):
        total = int(mask[row].sum().item())
        assert total == 32, f"Row {row}: expected exactly 32 masked cells, got {total}"
        assert int(mask[row, 0].sum().item()) == 0, f"Row {row}: first timestep should not be masked"
        found = 0
        for t in range(1, 4):
            for qy, qx in quadrants:
                block = mask[row, t, qy:qy+2, qx:qx+2]
                if block.all():
                    found += 1
                elif block.any():
                    raise AssertionError(f"Row {row} timestep {t} quadrant ({qy},{qx}): partial block; not contiguous")
        assert found > 0, f"Row {row}: at least one block should be sampled"



def test_distribution_gate_rejects_material_scale_mismatch() -> None:
    runner = _runner()
    runner._assert_distributions_compatible(_stats(), _stats(), "matching")
    with pytest.raises(ValueError, match="differs by"):
        runner._assert_distributions_compatible(_stats(), _stats(scale=10000.0), "unscaled")


def test_pool_matching_rejects_composite_bin_mismatch() -> None:
    runner = _runner()
    generic = {
        "geography_bins_10deg": {"+00:+000": 2},
        "geography_composite_bins": {"+00:+000|clear=80-090%|token=70-080%": 2},
    }
    agro = {
        "geography_bins_10deg": {"+00:+000": 2},
        "geography_composite_bins": {"+00:+000|clear=40-050%|token=70-080%": 2},
    }
    with pytest.raises(ValueError, match="composite matching"):
        runner._assert_pool_matching(generic, agro)


def test_pool_matching_rejects_missing_composite_keys() -> None:
    runner = _runner()
    missing = {"geography_bins_10deg": {"+00:+000": 2}}
    with pytest.raises(ValueError, match="composite_bins"):
        runner._assert_pool_matching(missing, {"geography_bins_10deg": {"+00:+000": 2}, "geography_composite_bins": {"x": 1}})


def test_model_forward_output_has_pred_and_mask_attrs() -> None:
    runner = _runner()
    device = torch.device("cpu")
    model = runner._make_model(runner.ARM_SPECS["A_pool_generic_fixed"], type("Shapes", (), {"shapes": type("S", (), {"s2_channels": 11, "s1_channels": 2})()})())
    batch = {
        "s2": torch.randn(2, 4, 11, 16, 16),
        "s1": torch.randn(2, 4, 2, 16, 16),
        "doy": torch.tensor([[30., 120., 210., 300.]]).repeat(2, 1),
        "s2_available": torch.ones(2, 4),
        "s1_available": torch.ones(2, 4),
        "s2_mask": torch.ones(2, 4, 16, 16, dtype=torch.uint8),
        "s1_mask": torch.ones(2, 4, 16, 16, dtype=torch.uint8),
    }
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 2:] = True
    output = model(target_mask=mask, **batch)
    assert hasattr(output, "local_pred")
    assert hasattr(output, "local_mask")


def test_mean_rows_grouping_preserves_checkpoint_role() -> None:
    runner = _runner()
    rows = [
        {"arm": "A", "model": "embedding", "holdout": "x", "condition": "clean", "label_budget": 1.0, "robustness_protocol": "clean", "seed": 42, "checkpoint_role": "best", "f1": 0.9, "auc": 0.8, "balanced_accuracy": 0.85, "calibrated_f1": 0.88, "calibrated_balanced_accuracy": 0.84},
        {"arm": "A", "model": "embedding", "holdout": "x", "condition": "clean", "label_budget": 1.0, "robustness_protocol": "clean", "seed": 42, "checkpoint_role": "final", "f1": 0.85, "auc": 0.78, "balanced_accuracy": 0.82, "calibrated_f1": 0.83, "calibrated_balanced_accuracy": 0.80},
    ]
    grouped = runner._mean_rows(rows, ["arm", "model", "holdout", "condition", "label_budget", "robustness_protocol", "checkpoint_role"])
    assert len(grouped) == 2
    assert grouped[0]["checkpoint_role"] == "best"
    assert grouped[1]["checkpoint_role"] == "final"
    assert abs(grouped[0]["mean_f1"] - 0.9) < 1e-6
    assert abs(grouped[1]["mean_f1"] - 0.85) < 1e-6


def test_validate_gpu_setup_rejects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A", "A", "B"], 1: ["C"]}, raising=False)
    with pytest.raises(ValueError, match="Duplicate"):
        runner._validate_gpu_setup(["A", "B", "C"])


def test_validate_gpu_setup_rejects_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        runner._validate_gpu_setup(["A_pool_generic_fixed"])


def test_validate_gpu_setup_rejects_insufficient_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="insufficient"):
        runner._validate_gpu_setup(["A_pool_generic_fixed"])


def test_validate_gpu_setup_rejects_missing_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    arms_with_unknown = list(runner.DEFAULT_ARMS) + ["Z_nonexistent"]
    with pytest.raises(ValueError, match="missing"):
        runner._validate_gpu_setup(arms_with_unknown)


def test_validate_gpu_setup_rejects_extra_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="extra"):
        runner._validate_gpu_setup(["A_pool_generic_fixed"])


def test_validate_gpu_setup_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    runner._validate_gpu_setup(runner.DEFAULT_ARMS)


def test_run_workers_immediate_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"], 1: ["B"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_fail_immediate, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A", "B"]
    with pytest.raises(RuntimeError, match="immediate failure"):
        runner._run_workers(arms, preprocess_qa)


def test_run_workers_sibling_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"], 1: ["B"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_sleep_then_fail, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A", "B"]
    with pytest.raises(RuntimeError, match="GPU 1 crashed"):
        runner._run_workers(arms, preprocess_qa)


def test_run_workers_large_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"], 1: ["B"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_large_payload, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A", "B"]
    results = runner._run_workers(arms, preprocess_qa)
    assert len(results) == 2
    for gpu_id, result_path in results.items():
        assert result_path.exists()
        payload = json.loads(result_path.read_text())
        assert len(payload) == 50  # 1 arm * 50 entries


def test_run_workers_all_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"], 1: ["B"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_ok, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A", "B"]
    results = runner._run_workers(arms, preprocess_qa)
    assert len(results) == 2
    for gpu_id, result_path in results.items():
        assert result_path.exists()


def test_run_workers_os_exit_before_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"], 1: ["B"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_os_exit_before_status, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A", "B"]
    with pytest.raises(RuntimeError, match="without reporting success"):
        runner._run_workers(arms, preprocess_qa)


def test_run_workers_os_exit_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_success_then_os_exit, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A"]
    with pytest.raises(RuntimeError, match="after reporting success"):
        runner._run_workers(arms, preprocess_qa)


def test_run_workers_hang_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_success_then_hang, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A"]
    with pytest.raises(RuntimeError, match="after reporting success"):
        runner._run_workers(arms, preprocess_qa)


def test_run_workers_happy_gpu_hangs_then_sibling_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    runner._MP_CONTEXT = "fork"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(runner, "GPU_ASSIGNMENTS", {0: ["A"], 1: ["B"]}, raising=False)
    monkeypatch.setattr(runner, "_run_worker", _test_worker_gpu0_hangs_gpu1_fails, raising=False)
    preprocess_qa = {"output_dir": str(tmp_path)}
    arms = ["A", "B"]
    with pytest.raises(RuntimeError, match="GPU 1 crashed"):
        runner._run_workers(arms, preprocess_qa)
