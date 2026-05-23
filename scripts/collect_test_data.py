# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datalad",
#     "pandas",
#     "pybids",
# ]
# ///

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import bids
import pandas as pd
from datalad import api

readme_template = """# PETPrep Test Data Collection

## Overview

This dataset contains a curated collection of PET imaging data from multiple
OpenNeuro datasets,compiled for testing and development of the PETPrep software pipeline.
The data has been selected to provide a diverse range of PET imaging scenarios for comprehensive
software testing.

## Dataset Information

- **Dataset Type**: Raw BIDS data
- **BIDS Version**: 1.7.0
- **License**: CC0 (Public Domain)
- **Compiled for**: PETPrep software testing and development

## Included Datasets

This collection includes data from the following OpenNeuro datasets:

{dataset_list}
## Data Structure

The dataset follows the Brain Imaging Data Structure (BIDS) specification:

```
├── dataset_description.json
├── participants.tsv
├── sub-*/                    # Subject directories
│   ├── anat/                 # Anatomical data
│   │   └── sub-*_T1w.nii.gz
│   └── pet/                  # PET data
│       ├── sub-*_pet.nii.gz
│       ├── sub-*_pet.json
│       └── sub-*_blood.tsv   # Blood data (if available)
```

## Usage

This dataset is intended for:
- PETPrep software testing and validation
- Development of PET preprocessing pipelines
- Educational purposes in PET data analysis

## Citation

If you use this test dataset, please cite:
- The original OpenNeuro datasets
- The PETPrep software: [PETPrep GitHub Repository](https://github.com/nipreps/petprep)

## Acknowledgments

- OpenNeuro for hosting the original datasets
- The BIDS community for data organization standards
- Contributors to the PETPrep project

## Contact

For questions about this test dataset or PETPrep:
- PETPrep GitHub: https://github.com/nipreps/petprep
- OpenNeuro: https://openneuro.org

---

*This is a test dataset compiled for software development purposes. Please refer to the original
 datasets for research use.*
"""


# Create dataset_description.json content
def create_dataset_description():
    """Create BIDS dataset_description.json content."""
    return {
        'Name': 'PETPrep Test Data Collection',
        'BIDSVersion': '1.7.0',
        'DatasetType': 'raw',
        'License': 'CC0',
        'Authors': ['datalad', 'python', 'make', 'openneuro'],
        'HowToAcknowledge': 'Please cite the original datasets and PETPrep software.',
        'Funding': [
            'This test data collection was created for PETPrep development and testing purposes'
        ],
        'EthicsApprovals': [
            'This is a test dataset compiled from publicly available BIDS datasets for software',
            'testing purposes',
        ],
        'ReferencesAndLinks': [
            'https://github.com/nipreps/petprep',
            'https://openneuro.org',
        ],
        'DatasetDOI': '10.18112/openneuro.ds000000.v1.0.0',
        'HEDVersion': '8.0.0',
    }


# Create README.md content
def create_readme_content(pet_datasets, readme_template):
    """Create README content dynamically based on the datasets."""

    # Generate dataset list dynamically
    dataset_list = ''
    for i, (dataset_id, meta) in enumerate(pet_datasets.items(), 1):
        dataset_list += f'{i}. **{dataset_id}**: {meta["description"]}\n'

    return readme_template.format(dataset_list=dataset_list)


DEFAULT_PET_DATASETS = {
    'ds005619': {
        'version': '1.1.0',
        'description': '[18F]SF51, a Novel 18F-labeled PET Radioligand for '
        'Translocator Protein 18kDa (TSPO) in Brain, Works Well '
        'in Monkeys but Fails in Humans',
        'subject_ids': ['sf02'],
    },
    'ds004868': {
        'version': '1.0.4',
        'description': '[11C]PS13 demonstrates pharmacologically selective and '
        'substantial binding to cyclooxygenase-1 (COX-1) in the '
        'human brain',
        'subject_ids': ['PSBB01'],
    },
    'ds004869': {
        'version': '1.1.1',
        'description': 'https://openneuro.org/datasets/ds004869/versions/1.1.1',
        'subject_ids': ['01'],
    },
}

OPENNEURO_TEMPLATE_STRING = 'https://github.com/OpenNeuroDatasets/{DATASET_ID}.git'


def download_test_data(
    working_directory: Path | None = None,
    output_directory: Path | None = None,
    pet_datasets_json: dict | None = None,  # Default to None, not the dict
    derivatives: list[str] | None = None,
):
    # Use default datasets if no JSON file provided
    if pet_datasets_json is None:
        datasets_to_use = DEFAULT_PET_DATASETS  # Use the default defined at module level
    else:
        # Load from JSON file
        with open(pet_datasets_json) as infile:
            datasets_to_use = json.load(infile)

    if derivatives is None:
        derivatives = []

    if not working_directory:
        working_directory = TemporaryDirectory()

    if not output_directory:
        output_directory = os.getcwd()

    with working_directory as data_path:
        combined_participants_tsv = pd.DataFrame()
        combined_subjects = []
        for (
            dataset_id,
            meta,
        ) in datasets_to_use.items():  # Use datasets_to_use instead of pet_datasets
            dataset_path = Path(data_path) / Path(dataset_id)
            if dataset_path.is_dir() and len(sys.argv) <= 1:
                dataset_path.rmdir()
            dataset = api.install(
                path=dataset_path,
                source=OPENNEURO_TEMPLATE_STRING.format(DATASET_ID=dataset_id),
            )
            # api.unlock(str(dataset_path))
            dataset.unlock()

            # see how pybids handles this datalad nonsense
            b = bids.layout.BIDSLayout(
                dataset_path,
                derivatives=False,
                validate=False,
            )  # when petderivatives are a thing, we'll think about using pybids to get them

            # Access participants.tsv
            participants_files = b.get(
                suffix='participants',
                extension='.tsv',
                return_type='file',
                scope='raw',
            )
            if participants_files:
                participants_file = participants_files[0]

                # Read participants.tsv as pandas DataFrame
                participants_df = pd.read_csv(participants_file, sep='\t')

                # Combine with overall participants DataFrame
                combined_participants_tsv = pd.concat(
                    [combined_participants_tsv, participants_df], ignore_index=True
                )
            # if a subset of subjects are specified collect only those subjects in the install
            if meta.get('subject_ids', []):
                for sid in meta['subject_ids']:
                    combined_subjects.append(sid)
                    # Get the entire subject directory content including git-annex files
                    subject_dir = dataset_path / f'sub-{sid}'
                    if not subject_dir.exists():
                        continue
                    # First, get all content in the subject directory
                    # (this retrieves git-annex files)
                    dataset.get(str(subject_dir))

                    # Then collect all files after they've been retrieved
                    all_files = []
                    for file_path in subject_dir.rglob('*'):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(dataset_path)
                            all_files.append(str(relative_path))

                    for deriv in derivatives:
                        print(f'Getting derivative: {deriv}/sub-{sid}')
                        deriv_dir = dataset_path / 'derivatives' / deriv / f'sub-{sid}'
                        try:
                            dataset.get(str(deriv_dir))
                        except Exception as e:  # noqa: BLE001
                            print(f'Error getting derivative {deriv}/sub-{sid}: {e}')
                            continue
                        for dv in deriv_dir.rglob('*'):
                            if dv.is_file():
                                relative_path = dv.relative_to(dataset_path)
                                all_files.append(str(relative_path))

                    # Copy all files to output directory
                    for f in all_files:
                        print(f)
                        # Unlock the file to make it writable
                        api.unlock(path=str(dataset_path / f), dataset=str(dataset_path))
                        source_file = dataset_path / f
                        relative_path = source_file.relative_to(dataset_path)
                        target_file = Path(output_directory) / relative_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, target_file)

            else:
                combined_subjects += b.get(return_type='id', target='subject')
                # Get all files first
                dataset.get(dataset_path)
                api.unlock(path=str(dataset_path), dataset=dataset)
                shutil.copytree(dataset_path, output_directory)

        combined_subjects = [f'sub-{s}' for s in combined_subjects]

        # Filter participants DataFrame to keep only subjects in combined_subjects list
        combined_participants = combined_participants_tsv[
            combined_participants_tsv['participant_id'].isin(combined_subjects)
        ]

        # Only write files if a specific download path was provided
        dataset_desc_path = Path(output_directory) / 'dataset_description.json'
        readme_path = Path(output_directory) / 'README.md'

        with open(dataset_desc_path, 'w') as f:
            json.dump(create_dataset_description(), f, indent=4)

        with open(readme_path, 'w') as f:
            f.write(create_readme_content(datasets_to_use, readme_template))
        combined_participants.to_csv(
            Path(output_directory) / 'participants.tsv', sep='\t', index=False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PETPrepTestDataCollector',
        description='Collects PET datasets from OpenNeuro.org and'
        'combines them into a single BIDS dataset using datalad and pandas',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--working-directory',
        '-w',
        default=TemporaryDirectory(),
        help='Working directory for downloading and combining datasets,'
        'defaults to a temporary directory.',
    )
    parser.add_argument(
        '--output-directory',
        '-o',
        default=Path.cwd(),
        help='Output directory of combined dataset,'
        'defaults where this script is called from, presently current working directory.',
    )
    parser.add_argument(
        '--derivatives',
        '-d',
        nargs='+',
        type=str,
        help='Additional derivatives to include alongside the BIDS data.',
    )
    parser.add_argument(
        '--datasets-json',
        '-j',
        type=str,
        default=None,
        help="""Use a custom json of datasets along
a subset of subjects can also be specified.
The default is structured like the following:

{
    "ds005619": {
        "version": "1.1.0",
        "description": "[description]",
        "subject_ids": ["sf02"]
        },
    "ds004868": {
        "version": "1.0.4",
        "description": "[description]",
        "subject_ids": ["PSBB01"]
        },
    "ds004869": {
        "version": "1.1.1",
        "description": "[description]",
        "subject_ids": ["01"]
        }
},""",
    )
    args = parser.parse_args()

    download_test_data(
        working_directory=args.working_directory,
        output_directory=args.output_directory,
        pet_datasets_json=args.datasets_json,  # This will be None if not provided
        derivatives=args.derivatives,
    )
