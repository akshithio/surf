from src.evals.evaluation import RAW_STAT_S2_IDXS
from src.datasets.ssl4eo import S2_BANDS


def test_raw_stat_s2_indices_select_reflectance_bands_once() -> None:
    assert [S2_BANDS[index] for index in RAW_STAT_S2_IDXS] == [
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
    ]
