import numpy as np
import pytest

from ..resampling import resample_series


def test_resample_series_rejects_mismatched_hmc_transforms():
    """HMC transform mappings must contain one affine per PET volume."""

    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    coordinates = np.indices(data.shape[:3], dtype=np.float32)

    with pytest.raises(ValueError, match='Head-motion transform count .* PET volumes'):
        resample_series(data, coordinates, [np.eye(4)], output_dtype=np.float32)
