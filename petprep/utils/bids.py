# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
"""Utilities to handle BIDS inputs."""
import json
import os
import sys
from pathlib import Path
from bids import BIDSLayout

from nipype import logging
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    OutputMultiObject,
    Str,
    SimpleInterface,
)
LOGGER = logging.getLogger("nipype.interface")


def write_bidsignore(deriv_dir):
    bids_ignore = (
        "*.html",
        "logs/",
        "figures/",  # Reports
        "*_xfm.*",  # Unspecified transform files
        "*.surf.gii",  # Unspecified structural outputs
        # Unspecified functional outputs
        "*_boldref.nii.gz",
        "*_bold.func.gii",
        "*_mixing.tsv",
        "*_timeseries.tsv",
    )
    ignore_file = Path(deriv_dir) / ".bidsignore"

    ignore_file.write_text("\n".join(bids_ignore) + "\n")


def write_derivative_description(bids_dir, deriv_dir):
    from ..__about__ import DOWNLOAD_URL, __version__

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': 'fMRIPrep - fMRI PREProcessing workflow',
        'BIDSVersion': '1.4.0',
        'DatasetType': 'derivative',
        'GeneratedBy': [
            {
                'Name': 'fMRIPrep',
                'Version': __version__,
                'CodeURL': DOWNLOAD_URL,
            }
        ],
        'HowToAcknowledge': 'Please cite our paper (https://doi.org/10.1038/s41592-018-0235-4), '
        'and include the generated citation boilerplate within the Methods '
        'section of the text.',
    }

    # Keys that can only be set by environment
    if 'FMRIPREP_DOCKER_TAG' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            "Type": "docker",
            "Tag": f"nipreps/fmriprep:{os.environ['FMRIPREP_DOCKER_TAG']}",
        }
    if 'FMRIPREP_SINGULARITY_URL' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            "Type": "singularity",
            "URI": os.getenv('FMRIPREP_SINGULARITY_URL'),
        }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        orig_desc = json.loads(fname.read_text())

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasets'] = [
            {'URL': f'https://doi.org/{orig_desc["DatasetDOI"]}', 'DOI': orig_desc['DatasetDOI']}
        ]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    Path.write_text(deriv_dir / 'dataset_description.json', json.dumps(desc, indent=4))


def validate_input_dir(exec_env, bids_dir, participant_label):
    # Ignore issues and warnings that should not influence FMRIPREP
    import subprocess
    import tempfile

    validator_config_dict = {
        "ignore": [
            "EVENTS_COLUMN_ONSET",
            "EVENTS_COLUMN_DURATION",
            "TSV_EQUAL_ROWS",
            "TSV_EMPTY_CELL",
            "TSV_IMPROPER_NA",
            "VOLUME_COUNT_MISMATCH",
            "BVAL_MULTIPLE_ROWS",
            "BVEC_NUMBER_ROWS",
            "DWI_MISSING_BVAL",
            "INCONSISTENT_SUBJECTS",
            "INCONSISTENT_PARAMETERS",
            "BVEC_ROW_LENGTH",
            "B_FILE",
            "PARTICIPANT_ID_COLUMN",
            "PARTICIPANT_ID_MISMATCH",
            "TASK_NAME_MUST_DEFINE",
            "PHENOTYPE_SUBJECTS_MISSING",
            "STIMULUS_FILE_MISSING",
            "DWI_MISSING_BVEC",
            "EVENTS_TSV_MISSING",
            "TSV_IMPROPER_NA",
            "ACQTIME_FMT",
            "Participants age 89 or higher",
            "DATASET_DESCRIPTION_JSON_MISSING",
            "FILENAME_COLUMN",
            "WRONG_NEW_LINE",
            "MISSING_TSV_COLUMN_CHANNELS",
            "MISSING_TSV_COLUMN_IEEG_CHANNELS",
            "MISSING_TSV_COLUMN_IEEG_ELECTRODES",
            "UNUSED_STIMULUS",
            "CHANNELS_COLUMN_SFREQ",
            "CHANNELS_COLUMN_LOWCUT",
            "CHANNELS_COLUMN_HIGHCUT",
            "CHANNELS_COLUMN_NOTCH",
            "CUSTOM_COLUMN_WITHOUT_DESCRIPTION",
            "ACQTIME_FMT",
            "SUSPICIOUSLY_LONG_EVENT_DESIGN",
            "SUSPICIOUSLY_SHORT_EVENT_DESIGN",
            "MALFORMED_BVEC",
            "MALFORMED_BVAL",
            "MISSING_TSV_COLUMN_EEG_ELECTRODES",
            "MISSING_SESSION",
        ],
        "error": ["NO_T1W"],
        "ignoredFiles": ['/dataset_description.json', '/participants.tsv'],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = set([s.name[4:] for s in bids_dir.glob('sub-*')])
        selected_subs = set([s[4:] if s.startswith('sub-') else s for s in participant_label])
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                'Data for requested participant(s) label(s) not found. Could '
                'not find data for participant(s): %s. Please verify the requested '
                'participant labels.'
            )
            if exec_env == 'docker':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the docker container. Please make sure all '
                    'volumes are mounted properly (see https://docs.docker.com/'
                    'engine/reference/commandline/run/#mount-volume--v---read-only)'
                )
            if exec_env == 'singularity':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the singularity container. Please make sure '
                    'all paths are mapped properly (see https://www.sylabs.io/'
                    'guides/3.0/user-guide/bind_paths_and_mounts.html)'
                )
            raise RuntimeError(error_msg % ','.join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict["ignoredFiles"].append("/sub-%s/**" % sub)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(['bids-validator', str(bids_dir), '-c', temp.name])
        except FileNotFoundError:
            print("bids-validator does not appear to be installed", file=sys.stderr)

def collect_data(
    bids_dir,
    participant_label,
    bids_validate=False,
    bids_filters=None,
):
    """
    Uses pybids to retrieve the input data for a given participant

    Examples
    --------
    >>> bids_root, _ = collect_data(str(datadir / 'ds054'), '100185',
    ...                             bids_validate=False)
    >>> bids_root['fmap']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/fmap/sub-100185_magnitude1.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_magnitude2.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_phasediff.nii.gz']
    >>> bids_root['bold']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_bold.nii.gz']
    >>> bids_root['sbref']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_sbref.nii.gz']
    >>> bids_root['t1w']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/anat/sub-100185_T1w.nii.gz']
    >>> bids_root['t2w']  # doctest: +ELLIPSIS
    []
    >>> bids_root, _ = collect_data(str(datadir / 'ds051'), '01',
    ...                             bids_validate=False, bids_filters={'t1w':{'run': 1}})
    >>> bids_root['t1w']  # doctest: +ELLIPSIS
    ['.../ds051/sub-01/anat/sub-01_run-01_T1w.nii.gz']

    """
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate)

    queries = {
        "flair": {"datatype": "anat", "suffix": "FLAIR"},
        "t2w": {"datatype": "anat", "suffix": "T2w"},
        "t1w": {"datatype": "anat", "suffix": "T1w"},
        "roi": {"datatype": "anat", "suffix": "roi"},
        "pet": {"suffix": "pet"}
    }
    bids_filters = bids_filters or {}
    for acq, entities in bids_filters.items():
        queries[acq].update(entities)


    subj_data = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                extension=["nii", "nii.gz"],
                **query,
            )
        )
        for dtype, query in queries.items()
    }

    return subj_data, layout

class _BIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    subject_data = traits.Dict(Str, traits.Any)
    subject_id = Str()


class _BIDSDataGrabberOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc="output data structure")
    fmap = OutputMultiObject(desc="output fieldmaps")
    bold = OutputMultiObject(desc="output functional images")
    sbref = OutputMultiObject(desc="output sbrefs")
    t1w = OutputMultiObject(desc="output T1w images")
    roi = OutputMultiObject(desc="output ROI images")
    t2w = OutputMultiObject(desc="output T2w images")
    flair = OutputMultiObject(desc="output FLAIR images")
    pet = OutputMultiObject(desc="output PET images")


class BIDSDataGrabber(SimpleInterface):
    """
    Collect files from a BIDS directory structure.

    >>> bids_src = BIDSDataGrabber(anat_only=False)
    >>> bids_src.inputs.subject_data = bids_collect_data(
    ...     str(datadir / 'ds114'), '01', bids_validate=False)[0]
    >>> bids_src.inputs.subject_id = '01'
    >>> res = bids_src.run()
    >>> res.outputs.t1w  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../ds114/sub-01/ses-retest/anat/sub-01_ses-retest_T1w.nii.gz',
     '.../ds114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz']

    """

    input_spec = _BIDSDataGrabberInputSpec
    output_spec = _BIDSDataGrabberOutputSpec
    _require_funcs = True

    def __init__(self, *args, **kwargs):
        anat_only = kwargs.pop("anat_only")
        anat_derivatives = kwargs.pop("anat_derivatives", None)
        super(BIDSDataGrabber, self).__init__(*args, **kwargs)
        if anat_only is not None:
            self._require_funcs = not anat_only
        self._require_t1w = anat_derivatives is None

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results["out_dict"] = bids_dict
        self._results.update(bids_dict)

        if self._require_t1w and not bids_dict['t1w']:
            raise FileNotFoundError(
                "No T1w images found for subject sub-{}".format(self.inputs.subject_id)
            )

        if self._require_funcs and not bids_dict["pet"]:
            raise FileNotFoundError(
                "No PET images found for subject sub-{}".format(
                    self.inputs.subject_id
                )
            )

        for imtype in ["pet"]:
            if not bids_dict[imtype]:
                LOGGER.info(
                    'No "%s" images found for sub-%s', imtype, self.inputs.subject_id
                )

        return runtime
