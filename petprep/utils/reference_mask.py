import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball, binary_dilation, binary_erosion


def generate_reference_region(
    seg_img: nib.Nifti1Image,
    config: dict,
    gm_probseg_img: nib.Nifti1Image | None = None,
) -> nib.Nifti1Image:
    """Generate a reference region using a flexible config.

    Config keys:
        - refmask_indices (list[int]): Labels to include in the reference region.
        - exclude_indices (list[int], optional): Labels to subtract (after dilation).
        - erode_by_voxels (int, optional): Number of voxels to erode the target region.
        - dilate_by_voxels (int, optional): Dilation radius for excluded regions.
        - smooth_fwhm_mm (float, optional): FWHM for smoothing the target region.
        - target_volume_ml (float, optional): Keep only the top N voxels by smoothed value.
        - gm_prob_threshold (float, optional): Threshold the final mask using a
          gray matter probability map. Requires ``gm_probseg_img``.

    Returns:
        nib.Nifti1Image: Final reference mask.
    """

    data = np.rint(seg_img.get_fdata()).astype(int)
    affine = seg_img.affine
    header = seg_img.header
    zooms = header.get_zooms()
    voxel_vol_mm3 = np.prod(zooms)

    # Step 1: Create binary target mask
    mask = np.isin(data, config['refmask_indices']).astype(np.uint8)

    # Step 2: Optional erosion
    if 'erode_by_voxels' in config and config['erode_by_voxels'] > 0:
        mask = binary_erosion(mask, ball(config['erode_by_voxels'])).astype(np.uint8)

    # Step 3: Optional exclusion
    if config.get('exclude_indices'):
        exclude = np.isin(data, config['exclude_indices'])  # bool mask
        if 'dilate_by_voxels' in config and config['dilate_by_voxels'] > 0:
            exclude = binary_dilation(exclude, ball(config['dilate_by_voxels']))
        mask[exclude] = 0

    # Step 4: Optional smoothing + volume constraint
    if 'smooth_fwhm_mm' in config and 'target_volume_ml' in config:
        sigma = config['smooth_fwhm_mm'] / (2.3548 * zooms[0])  # approximate sigma from FWHM
        smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)

        target_voxels = int((config['target_volume_ml'] * 1000) / voxel_vol_mm3)
        values = np.sort(smoothed[mask > 0].flatten())
        if target_voxels >= len(values):
            threshold = values[0]
        else:
            threshold = values[-target_voxels]
        mask = ((smoothed >= threshold) & (mask > 0)).astype(np.uint8)

    # Step 5: Optional gray matter probability thresholding
    if gm_probseg_img is not None and 'gm_prob_threshold' in config:
        gm_prob = gm_probseg_img.get_fdata()
        if gm_prob.shape != mask.shape:
            raise ValueError('gm_probseg_img does not match segmentation shape')
        mask = (mask > 0) & (gm_prob >= config['gm_prob_threshold'])
        mask = mask.astype(np.uint8)

    return nib.Nifti1Image(mask, affine, header)


def mask_to_stats(mask_file: str, mask_name: str) -> str:
    """Generate a TSV table of morphological statistics from a binary mask."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np
    import pandas as pd

    mask_file = Path(mask_file)
    img = nb.load(mask_file)
    data = img.get_fdata() > 0
    voxel_vol_mm3 = np.prod(img.header.get_zooms())
    volume_mm3 = float(data.sum() * voxel_vol_mm3)

    df = pd.DataFrame(
        {
            'index': [1],
            'name': [mask_name],
            'volume-mm3': [volume_mm3],
        }
    )

    out_file = mask_file.with_suffix('.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)
