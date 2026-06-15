# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Miscellaneous utilities."""

from functools import cache
from pathlib import Path


def check_deps(workflow):
    """Make sure dependencies are present in this system."""
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and which(node.interface._cmd.split()[0]) is None)
    )


def fips_enabled():
    """
    Check if FIPS is enabled on the system.

    For more information, see:
    https://github.com/nipreps/fmriprep/issues/2480#issuecomment-891199276
    """
    from pathlib import Path

    fips = Path('/proc/sys/crypto/fips_enabled')
    return fips.exists() and fips.read_text()[0] != '0'


def estimate_pet_mem_usage(pet_fname: str) -> tuple[int, dict]:
    """Estimate memory usage for a PET series."""
    pet_path = Path(pet_fname)
    stat = pet_path.stat()
    return _estimate_pet_mem_usage_cached(str(pet_path), stat.st_mtime_ns, stat.st_size)


@cache
def _estimate_pet_mem_usage_cached(
    pet_fname: str,
    _mtime_ns: int,
    _size: int,
) -> tuple[int, dict]:
    """Estimate memory usage for a PET series."""
    import nibabel as nb
    import numpy as np

    img = nb.load(pet_fname)
    nvox = int(np.prod(img.shape, dtype='u8'))
    spatial_nvox = int(np.prod(img.shape[:3], dtype='u8'))
    # Assume tools will coerce to 8-byte floats to be safe
    pet_size_gb = 8 * nvox / (1024**3)
    frame_size_gb = 8 * spatial_nvox / (1024**3)

    if img.ndim == 4:
        pet_tlen = img.shape[3]
    elif img.ndim == 3:
        pet_tlen = 1
    else:
        raise ValueError('PET image must be 3D or 4D')

    mem_gb = {
        'filesize': pet_size_gb,
        'frame': frame_size_gb,
        'reference': max(pet_size_gb * 1.5, pet_size_gb + frame_size_gb * 4),
        'resampled': pet_size_gb * 4,
        'largemem': pet_size_gb * (max(pet_tlen / 100, 1.0) + 4),
    }

    return pet_tlen, mem_gb


estimate_pet_mem_usage.cache_clear = _estimate_pet_mem_usage_cached.cache_clear
estimate_pet_mem_usage.cache_info = _estimate_pet_mem_usage_cached.cache_info
