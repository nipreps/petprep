"""CATNIP-inspired initialization and selection for direct PET normalization.

The principal-axis sign search is adapted from CATNIP by Granville Matheson.
Copyright (c) 2026 Granville Matheson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from pathlib import Path

import nibabel as nb
import numpy as np

REGISTRATION_MASK_DILATION_MM = 5.0


def registration_masks(fixed_mask, radius_mm):
    """Dilate the template mask in physical space and mask only the SyN stage."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np
    from scipy.ndimage import binary_dilation

    image = nb.load(fixed_mask)
    data = image.get_fdata()
    if image.ndim != 3 or not np.isfinite(data).all() or not np.any(data > 0):
        raise ValueError('The template brain mask must be a finite, nonempty 3D image.')
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError('The registration mask dilation radius must be finite and nonnegative.')

    # A physical sphere accounts for anisotropic, rotated, or sheared grids.
    linear = image.affine[:3, :3]
    extent = np.ceil(radius_mm * np.linalg.norm(np.linalg.inv(linear), axis=1)).astype(int)
    offsets = np.stack(
        np.meshgrid(*(np.arange(-n, n + 1) for n in extent), indexing='ij'), axis=-1
    )
    sphere = np.sum((offsets @ linear.T) ** 2, axis=-1) <= radius_mm**2 + 1e-8
    dilated = binary_dilation(data > 0, structure=sphere)
    header = image.header.copy()
    header.set_data_dtype('uint8')
    out = Path.cwd() / 'template_registration_mask.nii.gz'
    nb.Nifti1Image(dilated.astype('uint8'), image.affine, header).to_filename(out)
    return ['NULL', 'NULL', str(out)]


def _moments(image, mask=None):
    data = image.get_fdata()
    if image.ndim != 3 or not np.isfinite(data).all():
        raise ValueError('Registration references must be finite 3D images.')
    weights = np.maximum(data, 0)
    if mask is not None:
        weights *= mask
    indices = np.column_stack(np.nonzero(weights))
    values = weights[tuple(indices.T)]
    if not values.size or values.sum() <= 0:
        raise ValueError('Registration references need positive intensity support.')
    coords = nb.affines.apply_affine(image.affine, indices)
    centroid = np.average(coords, axis=0, weights=values)
    centered = coords - centroid
    covariance = (centered.T * values) @ centered / values.sum()
    return centroid, covariance


def _axes(covariance):
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values[0] <= 0 or np.min(-np.diff(values) / values[0]) < 1e-3:
        return None
    for col in range(3):
        if vectors[np.argmax(np.abs(vectors[:, col])), col] < 0:
            vectors[:, col] *= -1
    return vectors


def registration_initializations(fixed_image, moving_image, fixed_mask):
    """Write fixed-to-moving ITK transforms; omit indeterminate principal axes."""
    from itertools import product

    from petprep.utils.pet_registration import _axes, _moments

    fixed, moving, mask = (nb.load(path) for path in (fixed_image, moving_image, fixed_mask))
    if fixed.shape != mask.shape or not np.allclose(fixed.affine, mask.affine):
        raise ValueError('The fixed reference and brain mask must share a grid.')
    cf, vf = _moments(fixed, mask.get_fdata() > 0)
    cm, vm = _moments(moving)
    candidates = [('header', np.eye(3), np.zeros(3), np.zeros(3)), ('com', np.eye(3), cm - cf, cf)]
    af, am = _axes(vf), _axes(vm)
    if af is not None and am is not None:
        rotations = [(am * signs) @ af.T for signs in product((1, -1), repeat=3)]
        rotation = max((r for r in rotations if np.linalg.det(r) > 0), key=np.trace)
        candidates.append(('principalaxes', rotation, cm - cf, cf))

    flip = np.diag([-1.0, -1.0, 1.0])
    files, names = [], []
    for name, rotation, translation, center in candidates:
        parameters = np.r_[(flip @ rotation @ flip).ravel(), flip @ translation]
        out = Path.cwd() / f'{name}.tfm'
        out.write_text(
            '#Insight Transform File V1.0\n#Transform 0\n'
            'Transform: AffineTransform_double_3_3\nParameters: '
            + ' '.join(f'{v:.17g}' for v in parameters)
            + '\nFixedParameters: '
            + ' '.join(f'{v:.17g}' for v in flip @ center)
            + '\n'
        )
        files.append(str(out))
        names.append(name)
    return files, names


def select_registration(fixed_image, fixed_mask, warped_images, forward, inverse, names):
    """Compare candidates on one fixed mask; preserve scores and the selected pair."""
    import json

    fixed = nb.load(fixed_image)
    mask = nb.load(fixed_mask).get_fdata() > 0
    fixed_data = fixed.get_fdata()[mask]
    if not fixed_data.size or np.ptp(fixed_data) == 0:
        raise ValueError('The registration scoring mask must contain varying fixed intensities.')
    records = []
    for name, path in zip(names, warped_images, strict=True):
        image = nb.load(path)
        if image.shape != fixed.shape or not np.allclose(image.affine, fixed.affine):
            raise ValueError('Registration candidates must be evaluated on the same fixed grid.')
        values = image.get_fdata()[mask]
        score = None
        if np.isfinite(values).all() and np.ptp(values) > 0:
            joint, _, _ = np.histogram2d(fixed_data, values, bins=32)
            joint /= joint.sum()
            independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
            positive = joint > 0
            score = float(
                np.sum(joint[positive] * np.log(joint[positive] / independent[positive]))
            )
        records.append({'Initialization': name, 'MutualInformation': score})
    eligible = [i for i, row in enumerate(records) if row['MutualInformation'] is not None]
    if not eligible:
        raise ValueError('No registration candidate produced a usable warped reference.')
    selected = max(eligible, key=lambda i: records[i]['MutualInformation'])
    report = Path.cwd() / 'registration.json'
    report.write_text(
        json.dumps(
            {
                'Experimental': True,
                'Method': 'ANTs rigid + affine + SyN',
                'SelectedInitialization': names[selected],
                'Candidates': records,
                'Description': 'Selection by fixed-mask mutual information. Similarity is not '
                'a measure of anatomical accuracy; inspect registration overlays.',
            },
            indent=2,
        )
    )
    return forward[selected], inverse[selected], warped_images[selected], str(report)
