from pathlib import Path

import nitransforms as nt

from petprep.utils.transforms import load_transforms


def test_load_transforms_falls_back_to_itk_h5(monkeypatch, tmp_path):
    h5_file = tmp_path / 'from-T1w_to-MNI_mode-image_xfm.h5'
    h5_file.write_bytes(b'not-x5-test-double')
    expected = nt.Affine()
    calls = []

    def fake_load(path: Path, fmt='X5'):
        calls.append(fmt)
        if fmt == 'X5':
            raise TypeError('Input file is not in X5 format')
        return expected

    monkeypatch.setattr(nt.manip, 'load', fake_load)

    assert load_transforms([h5_file], [False]) is expected
    assert calls == ['X5', None]
