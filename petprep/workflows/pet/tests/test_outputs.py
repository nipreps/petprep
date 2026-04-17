import json
from pathlib import Path

import nibabel as nb
import numpy as np
from niworkflows.utils.testing import generate_bids_skeleton

from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..outputs import (
    init_ds_pet_native_wf,
    init_ds_petmask_wf,
    init_ds_petref_wf,
    init_ds_refmask_wf,
    init_ds_volumes_wf,
)


def _prep_bids(tmp_path: Path) -> Path:
    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    for p in bids_dir.rglob('*.nii.gz'):
        img.to_filename(p)
    return bids_dir


def test_datasink_datatype(tmp_path: Path):
    bids_dir = _prep_bids(tmp_path)
    out_dir = tmp_path / 'out'
    with mock_config(bids_dir=bids_dir):
        wf = init_ds_petref_wf(bids_root=bids_dir, output_dir=out_dir, desc='hmc')
        assert wf.get_node('ds_petref').inputs.datatype == 'pet'
        wf = init_ds_petmask_wf(output_dir=out_dir, desc='brain')
        assert wf.get_node('ds_petmask').inputs.datatype == 'pet'
        wf = init_ds_refmask_wf(output_dir=out_dir, ref_name='test')
        ref_node = wf.get_node('ds_refmask')
        assert ref_node.inputs.datatype == 'anat'
        assert ref_node.inputs.desc == 'ref'
        assert ref_node.inputs.label == 'test'
        wf = init_ds_pet_native_wf(
            bids_root=bids_dir,
            output_dir=out_dir,
            pet_output=True,
            all_metadata=[{}],
        )
        assert wf.get_node('ds_pet').inputs.datatype == 'pet'
        wf = init_ds_volumes_wf(
            bids_root=bids_dir,
            output_dir=out_dir,
            metadata={},
        )
        assert wf.get_node('ds_pet').inputs.datatype == 'pet'
        assert wf.get_node('ds_ref').inputs.datatype == 'pet'
        assert wf.get_node('ds_mask').inputs.datatype == 'pet'


def test_refmask_sources(tmp_path: Path):
    bids_dir = _prep_bids(tmp_path)
    out_dir = tmp_path / 'out'
    gm_file = bids_dir / 'sub-01' / 'anat' / 'sub-01_label-GM_probseg.nii.gz'
    seg_file = bids_dir / 'sub-01' / 'anat' / 'sub-01_desc-aparcaseg_dseg.nii.gz'
    refmask_file = tmp_path / 'refmask.nii.gz'

    img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    for path in (gm_file, seg_file, refmask_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        img.to_filename(path)

    t1_file = bids_dir / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'

    with mock_config(bids_dir=bids_dir):
        wf = init_ds_refmask_wf(output_dir=out_dir, ref_name='test')
        wf.base_dir = tmp_path / 'work'

        inputnode = wf.get_node('inputnode')
        inputnode.inputs.source_files = str(gm_file)
        inputnode.inputs.anat_sources = str(t1_file)
        inputnode.inputs.segmentation = str(seg_file)
        inputnode.inputs.refmask = str(refmask_file)

        wf.run()

        out_files = list(out_dir.rglob('*desc-ref*_mask.nii.gz'))
        assert out_files, 'Reference mask derivative was not generated'
        out_file = Path(out_files[0])
        metadata = json.loads(out_file.with_suffix('').with_suffix('.json').read_text())
        sources = metadata.get('Sources', [])

        assert any('label-GM_probseg' in src for src in sources)
        assert any('T1w' in src for src in sources)
        assert any('dseg' in src for src in sources)
        assert all('/pet/' not in src for src in sources)


def test_prepare_timing_parameters_and_psf_metadata():
    from nipype.interfaces.base import Undefined

    from ..outputs import build_pvc_metadata_dict, build_pvc_tacs_dict, prepare_timing_parameters

    timing = prepare_timing_parameters(
        {
            'VolumeTiming': [0, 10],
            'AcquisitionDuration': [10, 10],
            'InjectedRadioactivity': 120,
            'InjectedRadioactivityUnits': 'MBq',
            'Units': 'Bq/mL',
        }
    )
    assert timing == {
        'FrameTimesStart': [0, 10],
        'FrameDuration': [10, 10],
        'InjectedRadioactivity': 120,
        'InjectedRadioactivityUnits': 'MBq',
        'Units': 'Bq/mL',
    }

    assert build_pvc_metadata_dict(pvc_method='GTM', fwhm_x=3, fwhm_y=4.5, fwhm_z=5) == {
        'PVCMethod': 'GTM',
        'FWHM_x': 3.0,
        'FWHM_y': 4.5,
        'FWHM_z': 5.0,
    }
    assert build_pvc_metadata_dict(pvc_method='GTM', fwhm_x=Undefined, fwhm_y=4.5, fwhm_z=5) == {
        'PVCMethod': 'GTM',
        'FWHM_y': 4.5,
        'FWHM_z': 5.0,
    }

    assert build_pvc_tacs_dict(pvc_method=None) == {}
    assert build_pvc_tacs_dict(
        pvc_method='GTM',
        fwhm_x=3,
        fwhm_y=4.5,
        fwhm_z=5,
        software_name='PETPVC',
        software_version='1.2.3',
        command_line='petprep /bids /out participant',
    ) == {
        'PVCMethod': 'GTM',
        'FWHM_x': 3.0,
        'FWHM_y': 4.5,
        'FWHM_z': 5.0,
        'SoftwareName': 'petpvc',
        'SoftwareVersion': '1.2.3',
        'CommandLine': 'petprep /bids /out participant',
    }

    assert build_pvc_tacs_dict(
        pvc_method='MG',
        fwhm_x=None,
        fwhm_y=4.5,
        fwhm_z=None,
        software_name='',
        software_version='',
        command_line='',
    ) == {
        'PVCMethod': 'MG',
        'FWHM_y': 4.5,
    }


def test_init_func_fit_reports_wf_with_atlas_and_refmask(tmp_path: Path):
    from ..outputs import init_func_fit_reports_wf

    wf = init_func_fit_reports_wf(
        freesurfer=True,
        output_dir=str(tmp_path / 'out'),
        ref_name='cerebellum',
        atlas_name='HOCPA',
    )

    node_names = wf.list_node_names()
    assert 'ds_report_refmask' in node_names
    assert 'atlas_overlay_report' in node_names
    assert 'ds_atlas_overlay' in node_names

    edge = wf._graph.get_edge_data(
        wf.get_node('crop_petref_atlas'), wf.get_node('atlas_overlay_report')
    )
    assert ('out_file', 'segmentation') in edge['connect']

    ds_refmask = wf.get_node('ds_pet_t1_refmask_report')
    assert ds_refmask.inputs.label == 'cerebellum'


def test_init_func_fit_reports_wf_without_optional_reports(tmp_path: Path):
    from ..outputs import init_func_fit_reports_wf

    wf = init_func_fit_reports_wf(
        freesurfer=False,
        output_dir=str(tmp_path / 'out'),
        ref_name='',
        atlas_name=None,
    )

    node_names = wf.list_node_names()
    assert 'ds_report_refmask' not in node_names
    assert 'atlas_overlay_report' not in node_names
