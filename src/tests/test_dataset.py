from pathlib import Path

import numpy as np

from src.datasets.dataset import MultimodalPatchDataset
from src.datasets.ssl4eo import _create_array, _open_output_group

def test_reflectance_divisor_leaves_ndvi_unchanged(tmp_path: Path) -> None:
    root = _open_output_group(tmp_path / "dataset.zarr", mode="w")
    _create_array(root, "s2", shape=(1, 1, 2, 1, 1), chunks=(1, 1, 2, 1, 1), dtype="float32")[:] = 0
    root["s2"][0, 0, :, 0, 0] = np.asarray([2500.0, 0.75], dtype=np.float32)
    _create_array(root, "s1", shape=(1, 1, 1, 1, 1), chunks=(1, 1, 1, 1, 1), dtype="float32")[:] = 0
    _create_array(root, "s2_mask", shape=(1, 1, 1, 1), chunks=(1, 1, 1, 1), dtype="uint8")[:] = 1
    _create_array(root, "s1_mask", shape=(1, 1, 1, 1), chunks=(1, 1, 1, 1), dtype="uint8")[:] = 1
    _create_array(root, "time", shape=(1,), chunks=(1,), dtype="S16")[:] = np.asarray(["2000-01-15"], dtype="S16")
    _create_array(root, "s2_bands", shape=(2,), chunks=(2,), dtype="S32")[:] = np.asarray(["B2", "NDVI"], dtype="S32")
    dataset = MultimodalPatchDataset(tmp_path / "dataset.zarr", s2_reflectance_divisor=10000.0)
    sample = dataset[0]
    assert float(sample["s2"][0, 0, 0, 0]) == 0.25
    assert float(sample["s2"][0, 1, 0, 0]) == 0.75
