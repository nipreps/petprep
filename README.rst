*PETPrep*: A Robust Preprocessing Pipeline for PET Data
=========================================================
*PETPrep* is a *NiPreps (NeuroImaging PREProcessing toolS)* application
(`www.nipreps.org <https://www.nipreps.org>`__) for the preprocessing of
positron emission tomography (PET) imaging.

.. image:: https://github.com/nipreps/petprep/actions/workflows/tests.yml/badge.svg
  :target: https://github.com/nipreps/petprep/actions/workflows/tests.yml
  :alt: Stable tests

.. image:: https://readthedocs.org/projects/petprep/badge/?version=latest
  :target: https://petprep.org/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/docker-nipreps/petprep-brightgreen.svg?logo=docker&style=flat
  :target: https://hub.docker.com/r/nipreps/petprep/tags/
  :alt: Docker image available!

.. image:: https://chanzuckerberg.github.io/open-science/badges/CZI-EOSS.svg
  :target: https://czi.co/EOSS
  :alt: CZI's Essential Open Source Software for Science

About
-----
.. image:: https://github.com/nipreps/petprep/blob/main/docs/_static/petprep-0.0.1.svg

*PETPrep* is a positron emission tomography (PET) data
preprocessing pipeline that is designed to provide an easily accessible,
state-of-the-art interface that is robust to variations in scan acquisition
protocols and that requires minimal user input, while providing easily
interpretable and comprehensive error and output reporting.
It performs comprehensive processing steps—such as motion correction,
segmentation, registration, partial volume correction, and extraction of time
activity curves, providing outputs that can be
easily submitted to a variety of group level analyses, pharmacokinetic modelling, 
graph theory measures, and surface or volume-based statistics.

.. note::

   *PETPrep* performs minimal preprocessing.
   Here we define 'minimal preprocessing'  as motion correction, generation of a 3D PET reference image, 
   normalization, and brain mask extraction.
   See the `workflows section of our documentation
   <https://petprep.readthedocs.io/en/latest/workflows.html>`__ for more details.

The *PETPrep* pipeline uses a combination of tools from well-known software
packages, including FSL_, ANTs_, FreeSurfer_, AFNI_, PETSurfer_ and PETPVC_.
This pipeline was designed to provide the best software implementation for each
state of preprocessing, and will be updated as newer and better neuroimaging
software become available.

This tool allows you to easily do the following:

- Take PET data from raw to fully preprocessed form.
- Apply motion correction to minimize artifacts from subject movement.
- Segment anatomy for improved regional characterization.
- Register data within and across subjects, and to template spaces.
- Perform partial volume correction.
- Extract time activity curves for pharmacokinetic modelling.
- Implement tools from different software packages.
- Achieve optimal data processing quality by using the best tools available.
- Generate preprocessing quality reports, with which the user can easily
  identify outliers.
- Receive verbose output concerning the stage of preprocessing for each
  subject, including meaningful errors.
- Automate and parallelize processing steps, which provides a significant
  speed-up from manual processing or shell-scripted pipelines.

PETPrep also extracts regional time-activity curves as tabular files with frame
timings and uptake values. These tables can be fed directly into
pharmacokinetic modeling tools such as kinfitr_ or PMOD_ to estimate tracer kinetics or compute binding estimates.

More information and documentation can be found at
https://petprep.readthedocs.io/

Principles
----------
*PETPrep* is built around three principles:

1. **Robustness** - The pipeline adapts the preprocessing steps depending on
   the input dataset and should provide results as good as possible
   independently of scanner make and scanning parameters.
2. **Ease of use** - Thanks to dependence on the BIDS standard (BIDS_), manual
   parameter input is reduced to a minimum, allowing the pipeline to run in an
   automatic fashion.
3. **"Glass box"** philosophy - Automation should not mean that one should not
   visually inspect the results or understand the methods.
   Thus, *PETPrep* provides visual reports for each subject, detailing the
   accuracy of the most important processing steps.
   This, combined with the documentation, can help researchers to understand
   the process and decide which subjects should be kept for the group level
   analysis.

Citation
--------
**Citation boilerplate**.
Please acknowledge this work using the citation boilerplate that *PETPrep* includes
in the visual report generated for every subject processed.
For a more detailed description of the citation boilerplate and its relevance,
please check out the
`NiPreps documentation <https://www.nipreps.org/intro/transparency/#citation-boilerplates>`__.

**Plagiarism disclaimer**.
The boilerplate text is public domain, distributed under the
`CC0 license <https://creativecommons.org/publicdomain/zero/1.0/>`__,
and we recommend *PETPrep* users to reproduce it verbatim in their works.
Therefore, if reviewers and/or editors raise concerns because the text is flagged by automated
plagiarism detection, please refer them to the *NiPreps* community and/or the note to this
effect in the `boilerplate documentation page <https://www.nipreps.org/intro/transparency/#citation-boilerplates>`__.

**Papers**.
*PETPrep* contributors have published the relevant papers:

**Other**.

License information
-------------------
*PETPrep* adheres to the
`general licensing guidelines <https://www.nipreps.org/community/licensing/>`__
of the *NiPreps framework*.

License
~~~~~~~
Copyright (c) the *NiPreps* Developers.

As of the 0.0.1 release series, *PETPrep* is
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
`http://www.apache.org/licenses/LICENSE-2.0
<http://www.apache.org/licenses/LICENSE-2.0>`__.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Acknowledgements
----------------
This work is steered and maintained by the `NiPreps Community <https://www.nipreps.org>`__.
This was supported by the BRAIN Initiative
grant (OpenNeuroPET, grant ID 1R24MH120004-01A1); the Novo Nordisk Foundation (OpenNeuroPET, grant ID NN20OC0063277); the Laura and John Arnold Foundation,
the NIH (grant NBIB R01EB020740, PI: Ghosh);
and NIMH (R24MH114705, R24MH117179, R01MH121867, PI: Poldrack)

.. _FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki
.. _ANTs: http://stnava.github.io/ANTs/
.. _FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
.. _AFNI: https://afni.nimh.nih.gov/
.. _PETSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/PetSurfer
.. _PETPVC: https://github.com/UCL/PETPVC
.. _kinfitr: https://github.com/mathesong/kinfitr
.. _PMOD: https://www.pmod.com/
.. _BIDS: https://bids.neuroimaging.io/