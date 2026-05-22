.. include:: links.rst

================================
FAQ - Frequently Asked Questions
================================

.. contents::
    :local:
    :depth: 1

------------

Should I run quality control of my images before running *PETPrep*?
--------------------------------------------------------------------
Yes. You should do so before any processing/analysis takes place.

Oftentimes (more often than we would like), images have fatal artifacts and problems.

Some exclusion criteria for data quality should be pre-specified before QC and any screening
of the original data.
Those exclusion criteria must be designed in agreement with the goals and challenges of the
experimental design.
For instance, when it is planned to run some cortical thickness analysis, images should be excluded
even when they present the most subtle ghosts or other artifacts that may introduce biases in surface
reconstruction.
However, if the same artifactual data was planned to be used just as a reference for spatial
normalization, some of those artifacts should be noted, but may not grant exclusion of the data.

When using publicly available datasets, an additional concern is that images may have gone through
some kind of preprocessing (see next question).


What if I find some images have undergone some pre-processing already (e.g., my T1w image is already skull-stripped)?
---------------------------------------------------------------------------------------------------------------------
These images imply an unknown level of preprocessing (e.g. was it already bias-field corrected?),
which makes it difficult to decide on best-practices for further processing.
*PETPrep* uses the anatomical image to support PET-to-anatomical registration,
segmentation, optional partial volume correction, reference-region masks, and
time-activity curve extraction.
If the T1w image has already been skull-stripped or otherwise modified, these
steps may become less reproducible because *PETPrep* can no longer control the
full anatomical preprocessing workflow.

For user-supplied data, we recommend reverting to the original, defaced T1w
images whenever possible.
If only preprocessed anatomical images are available, document the upstream
processing carefully and inspect the PET-to-anatomical registration, segmentation,
and derived TACs before including those subjects in group analyses.


My *PETPrep* run is hanging...
-------------------------------
When running on Linux platforms (or containerized environments, because they are built around
Ubuntu), there is a Python bug that affects *PETPrep* that drives the Linux kernel to kill
processes as a response to running out of memory.
Depending on the process killed by the kernel, *PETPrep* may crash with a ``BrokenProcessPool``
error or hang indefinitely, depending on settings.
While we are working on finding a solution that does not run up against this bug, this may take some
time.
This can be most easily resolved by allocating more memory to the process, if possible.

Please find more information regarding this error from discussions on
`NeuroStars <https://neurostars.org/tags/petprep>`_:

 * `memory issue when processing large amount of data <https://neurostars.org/t/memory-issue-when-processing-large-amount-of-data/2562>`_
 * `RAM CPUs reasonable to run pipelines like PETPrep <https://neurostars.org/t/how-much-ram-cpus-is-reasonable-to-run-pipelines-like-petprep/1086>`_
 * `memory allocation issues with PETPrep, Singularity and HPC <https://neurostars.org/t/memory-allocation-issues-with-petprep-singularity-and-hpc/2759>`_
 * `PETPrep v1.0.12 hanging <https://neurostars.org/t/petprep-v1-0-12-hanging/1661>`_.

Additionally, consider using the ``--low-mem`` flag, which will make some memory optimizations at the cost of disk space in the working directory.

I have already run ``recon-all`` on my subjects, can I reuse my outputs?
------------------------------------------------------------------------
Yes, as long as the FreeSurfer_ version previously used was ``6.0.0`` or newer.
If running with FreeSurfer, *PETPrep* checks if the output directory contains a ``freesurfer``
directory and reuses the outputs found.
Alternatively, you can use the ``--fs-subjects-dir`` flag to specify a different location for the existing FreeSurfer outputs.

ERROR: it appears that ``recon-all`` is already running
-------------------------------------------------------
When running FreeSurfer's ``recon-all``, an error may say *it appears it is already running*.
FreeSurfer creates files (called ``IsRunning.{rh,lh,lh+rh}``, under the ``scripts/`` folder)
to determine whether it is already executing ``recon-all`` on that particular subject
in another process, compute node, etc.
If a FreeSurfer execution terminates abruptly, those files are not wiped out, and therefore,
the next time you try to execute ``recon-all``, FreeSurfer *thinks* it is still running.
The output you get from PETPrep will contain something like: ::

  RuntimeError: Command:
  recon-all -autorecon2-volonly -openmp 8 -subjid sub-020 -sd /outputs/freesurfer -nogcareg -nocanorm -nocareg -nonormalization2 -nomaskbfs -nosegmentation -nofill
  Standard output:
  Subject Stamp: freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a
  Current Stamp: freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a
  INFO: SUBJECTS_DIR is /outputs/freesurfer
  Actual FREESURFER_HOME /opt/freesurfer
  -rw-rw-r-- 1 11239 users 207798 Apr  1 16:19 /outputs/freesurfer/sub-020/scripts/recon-all.log
  Linux 62324c0da859 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux

  ERROR: it appears that recon-all is already running
  for sub-020 based on the presence of /outputs/freesurfer/sub-020/scripts/IsRunning.lh+rh. It could
  also be that recon-all was running at one point but
  died in an unexpected way. If it is the case that there
  is a process running, you can kill it and start over or
  just let it run. If the process has died, you should type:

  rm /outputs/freesurfer/sub-020/scripts/IsRunning.lh+rh

  and re-run. Or you can add -no-isrunning to the recon-all
  command-line. The contents of this file are:
  ----------------------------------------------------------
  ------------------------------
  SUBJECT sub-020
  HEMI    lh rh
  DATE Fri Mar 22 20:33:09 UTC 2019
  USER root
  HOST 622795a21a5f
  PROCESSID 55530
  PROCESSOR x86_64
  OS Linux
  Linux 622795a21a5f 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
  $Id: recon-all,v 1.580.2.16 2017/01/18 14:11:24 oesteban Exp $
  ----------------------------------------------------------
  Standard error:

  Return code: 1

As suggested by the ``recon-all`` output message, deleting these files will enable
FreeSurfer to execute ``recon-all`` again.
In general, please be cautious of deleting files and mindful why a file may exist.

Running subjects in parallel
----------------------------
The recommended way to parallelize *PETPrep* is to run one subject per container
or batch job, with a shared read-only BIDS input directory and a unique working
directory for each job.
This keeps Nipype's intermediate files isolated and avoids multiple processes
writing to the same workflow cache.

When running many subjects at the same time, make sure that each job either
writes to a separate output directory that will be merged later, or that your
execution environment prevents two jobs from processing the same subject into
the same output tree at once.
Shared FreeSurfer derivatives may be reused with ``--fs-subjects-dir``, but they
should not be generated simultaneously by multiple jobs for the same subject.

How much CPU time and RAM should I allocate for a typical *PETPrep* run?
-------------------------------------------------------------------------
The recommended way to run *PETPrep* is to process one subject per container
instance.
A typical preprocessing run without surface processing with FreeSurfer can be
completed in about 2 hours with 4 CPUs or in about 1 hour with 16 CPUs.
More than 16 CPUs do not usually translate into faster processing times for a
single subject.
About 8GB of memory should be available for a single subject preprocessing run.

Runtime and memory use depend on the number of PET frames, whether FreeSurfer is
run or reused, whether surface or CIFTI outputs are requested, which
segmentation is selected with ``--seg``, and whether partial volume correction or
reference-region TAC extraction is enabled.
For larger dynamic acquisitions, atlas segmentations, or PVC workflows, allocate
additional memory and inspect the first few subjects before scaling out to the
whole dataset.

Below are some benchmark data that have been computed on a high performance cluster compute node with Intel E5-2683 v4
CPUs and 64 GB of physical memory:

.. figure:: _static/petprep_benchmark.svg

**Compute Time**: time in hours to complete the preprocessing for all subjects. **Physical Memory**: the maximum of RAM usage
used across all *PETPrep* processes as reported by the HCP job manager. **Virtual Memory**: the maximum of virtual memory used
across all *PETPrep* processes as reported by the HCP job manager. **Threads**: the maximum number of threads per process as
specified with ``–omp-nthreads`` in the PETPrep command.

The above figure illustrates that processing 2 subjects in 2 *PETPrep* instances with 8 CPUs each is approximately as fast as
processing 2 subjects in one *PETPrep* instance with 16 CPUs. However, on a distributed compute cluster, the two 8 CPU
instances may be allocated faster than the single 16 CPU instance, thus completing faster in practice. If more than one
subject is processed in a single *PETPrep* instance, then limiting the number of threads per process to roughly the
number of CPUs divided by the number of subjects is most efficient.

.. _upgrading:

A new version of *PETPrep* has been published, when should I upgrade?
----------------------------------------------------------------------
We follow a philosophy of releasing very often, although the pace is slowing down
with the maturation of the software.
It is very likely that your version gets outdated over the extent of your study.
If that is the case (an ongoing study), then we discourage changing versions.
In other words, **the whole dataset should be processed with the same version (and
same container build if they are being used) of *PETPrep*.**

On the other hand, if the project is about to start, then we strongly recommend
using the latest version of the tool.

In any case, if you can find your release listed as *flagged* in `this file
of our repo <https://github.com/nipreps/petprep/blob/master/.versions.json>`__,
then please update as soon as possible.

I'm running *PETPrep* via Singularity containers - how can I troubleshoot problems?
------------------------------------------------------------------------------------
We have extended `this documentation <singularity.html>`__ to cover some of the most
frequent issues other Singularity users have been faced with.
Generally, users have found it hard to `get TemplateFlow and Singularity to work
together <singularity.html#singularity-tf>`__.

What is *TemplateFlow* for?
---------------------------
*TemplateFlow* enables *PETPrep* to generate preprocessed outputs spatially normalized to
a number of different neuroimaging templates (e.g. MNI).
For further details, please check `its documentation section <spaces.html#templateflow>`__.

.. _tf_no_internet:

How do you use *TemplateFlow* in the absence of access to the Internet?
-----------------------------------------------------------------------
This is a fairly common situation in :abbr:`HPCs (high-performance computing)`
systems, where the so-called login nodes have access to the Internet but
compute nodes are isolated, or in PC/laptop environments if you are traveling.
*TemplateFlow* will require Internet access the first time it receives a
query for a template resource that has not been previously accessed.
If you know what are the templates you are planning to use, you could
prefetch them using the Python client.
In addition to the ``--output-spaces`` that you specify, *PETPrep* will
internally require the ``MNI152NLin2009cAsym`` template.
If the ``--skull-strip-template`` option is not set, then ``OASIS30ANTs``
will be used.
Finally, the ``--cifti-output`` argument requires ``MNI152NLin6Asym``.
Atlas-based segmentations selected with ``--seg`` may also add the corresponding
atlas template to the workflow.
To do so, follow the next steps.

1. By default, a mirror of *TemplateFlow* to store the resources will be
   created in ``$HOME/.cache/templateflow``.
   You can modify such a configuration with the ``TEMPLATEFLOW_HOME``
   environment variable, e.g.::

     $ export TEMPLATEFLOW_HOME=$HOME/.templateflow

2. Install the client within your favorite Python 3 environment (this can
   be done in your login-node, or in a host with Internet access,
   without need for Docker/Singularity)::

     $ python -m pip install -U templateflow

3. Use the ``get()`` utility of the client to pull down all the templates you'll
   want to use. For example::

     $ python -c "from templateflow.api import get; get(['MNI152NLin2009cAsym', 'MNI152NLin6Asym', 'OASIS30ANTs', 'MNIPediatricAsym', 'MNIInfant'])"

After getting the resources you'll need, you will just need to make sure your
runtime environment is able to access the filesystem, at the location of your
*TemplateFlow home* directory.
If you are an Apptainer (formerly Singularity) user, please check out `TemplateFlow and Singularity
<https://www.nipreps.org/apps/singularity/#templateflow-and-singularity>`__.

How do I select only certain files to be input to *PETPrep*?
-------------------------------------------------------------
For common PET selections, start with the dedicated command-line options:
``--participant-label``, ``--session-label``, ``--tracer-label``,
``--rec-label``, ``--run-label``, and ``--task-id``.
These options are usually enough when you want to process a subset of subjects,
sessions, tracers, reconstructions, runs, or tasks.
They are also the safest first choice when selecting PET inputs for a
longitudinal or multi-tracer study because the resulting command line remains
easy to audit.

For more specific selections, use the ``--bids-filter-file`` flag.
You can pass *PETPrep* a JSON file that
describes a custom BIDS filter for selecting files with PyBIDS, with the syntax
``{<query>: {<entity>: <filter>, ...},...}``. For example::

  {
      "t1w": {
          "datatype": "anat",
          "session": "02",
          "acquisition": null,
          "suffix": "T1w"
      },
      "pet": {
          "datatype": "pet",
          "session": "02",
          "suffix": "pet"
      }
  }

PETPrep uses the following queries, by default::

  {
    "fmap": {"datatype": "fmap"},
    "pet": {"datatype": "pet", "suffix": "pet"},
    "sbref": {"datatype": "func", "suffix": "sbref"},
    "flair": {"datatype": "anat", "suffix": "FLAIR"},
    "t2w": {"datatype": "anat", "suffix": "T2w"},
    "t1w": {"datatype": "anat", "suffix": "T1w"},
    "roi": {"datatype": "anat", "suffix": "roi"}
  }

Only modifications of these queries will have any effect. You may filter on any entity defined
in the PyBIDS
`config file <https://github.com/bids-standard/pybids/blob/master/bids/layout/config/bids.json>`__.
To select images that do not have the entity set, use json value: ``null``.
To select images that have any non-empty value for an entity use string: ``'*'``

If the anatomical inputs also need to be restricted, for example to process each
session in a longitudinal study with its own T1w image, include both the ``pet``
and ``t1w`` queries in the filter file.
This prevents a session-specific PET run from accidentally sharing anatomical
inputs from other sessions.
For longitudinal datasets where every selected session should be processed with
its own anatomical reference, prefer
``--subject-anatomical-reference sessionwise``.
That option applies session-specific filtering to both PET and anatomical inputs
without requiring a separate filter file for the common per-session case.

Which *PETPrep* options should I decide before processing my whole study?
-------------------------------------------------------------------------
Decide the analysis-defining options before running the full dataset, and keep
them fixed across subjects whenever possible.
In particular, choose the PET reference strategy (``--petref``), PET-to-anatomy
registration backend and degrees of freedom (``--pet2anat-method`` and
``--pet2anat-dof``), segmentation (``--seg``), reference-region mask options
(``--ref-mask-name`` and ``--ref-mask-index``), partial volume correction
settings (``--pvc-tool``, ``--pvc-method``, and ``--pvc-psf``), and output spaces
(``--output-spaces`` and, if needed, CIFTI outputs).
For longitudinal or multi-session studies, also decide whether sessions should
share a subject-level anatomical reference, use an unbiased subject-level
template, or be processed with session-specific anatomical references.
This is controlled by ``--subject-anatomical-reference``.

These choices determine the derivatives available for downstream analysis,
including regional TACs, reference-region TACs, PVC-corrected data, and
surface- or template-space outputs.
Running a small pilot subset first is often the most efficient way to confirm
registration quality, segmentation suitability, and model-ready outputs before
committing compute time to a full study.

Which *PETPrep* outputs do I need for kinetic modeling?
-------------------------------------------------------
*PETPrep* prepares PET data for downstream analysis; it does not estimate kinetic
model parameters itself.
For regional kinetic modeling, request the segmentation and reference-region
outputs that match your analysis plan.
The `segmentation <usage.html#segmentation>`__ options determine which regional
time-activity curves (TACs) are extracted, and the
`reference region mask <usage.html#reference-region-masks>`__ options can
additionally write TACs for a reference region.
The resulting ``*_desc-preproc_tacs.tsv`` files are documented in
`Time activity curves <outputs.html#time-activity-curves>`__.

When planning a modeling workflow, decide the segmentation, reference region,
partial volume correction, and output spaces before processing the full dataset.
Those choices affect which PET derivatives are produced and help keep the
preprocessing and modeling stages reproducible across subjects.

How should I choose ``--seg``, PVC, and reference masks for kinetic modeling?
-----------------------------------------------------------------------------
Choose these options together, starting from the model and regions you plan to
fit.
The ``--seg`` option determines the anatomical labels used for regional TAC
extraction, reference-region masks, and, when requested, partial volume
correction.
The default ``gtm`` segmentation is the most general FreeSurfer-based choice,
while atlas or region-specific segmentations should be selected when needed for
the specific analysis.

Reference masks should match the selected segmentation.
Predefined masks such as ``cerebellum``, ``neocortex`` and ``thalamus`` are
defined for ``--seg gtm``, while ``semiovale`` is defined for ``--seg wm``.
For custom reference regions, provide both ``--ref-mask-name`` and
``--ref-mask-index`` using label indices from the corresponding
``*_seg-<segmentation>_morph.tsv`` file.
This keeps the reference-region TAC tied to the same label space as the regional
TACs.

PVC choices should also be compatible with the segmentation.
``petsurfer`` PVC requires ``--seg gtm``; ``petpvc`` can use any *PETPrep*
segmentation.
The ``--pvc-tool``, ``--pvc-method`` and ``--pvc-psf`` options must be supplied
together, and the point spread function should be fixed across subjects and
sessions that will be compared.
When PVC is enabled, downstream PET derivatives and TACs are generated from the
PVC-corrected series and include the ``pvc-<method>`` entity where applicable.

In practice, run a small pilot subset with the planned segmentation, reference
mask and PVC settings before processing the full cohort.
Inspect registration, segmentation labels, reference masks, and TAC tables
together; a technically valid segmentation is not necessarily the right one for
the kinetic model.

I have now processed my data using *PETPrep*. How do I perform kinetic modeling?
--------------------------------------------------------------------------------
For regional analyses, PETFit_ can ingest *PETPrep* outputs and use the regional
TACs generated by *PETPrep* for kinetic modeling.
This is the recommended next step when your model is defined over regions from a
segmentation or atlas and you want to work with tabular TAC outputs.

If you want to estimate parametric images or run surface-based analyses, consider
PETSurfer-BIDS_.
PETSurfer-BIDS is built on top of *PETPrep* and uses *PETPrep* for the
preprocessing steps, while PETSurfer-BIDS handles voxel-wise smoothing, kinetic
modeling, and group analysis in ROI, volume, and surface spaces.

Can I use *PETPrep* outputs for parametric images or surface-based analyses?
----------------------------------------------------------------------------
Yes, but those analyses are usually run in a downstream tool rather than inside
*PETPrep* itself.
Use *PETPrep* to generate the motion-corrected, co-registered, optionally
partial-volume-corrected PET derivatives and the anatomical or surface
derivatives required by the downstream analysis.
PETSurfer-BIDS_ is the closest fit when the next step is voxel-wise kinetic
modeling, parametric imaging, or surface-based PET analysis.

Can *PETPrep* continue to run after encountering an error?
-----------------------------------------------------------
(Context: `#1756 <https://github.com/nipreps/petprep/issues/1756>`__)
Yes, although it requires access to previously computed intermediate results.
*PETPrep* is built on top of Nipype_, which uses a temporary folder to store the interim
results of the workflow.
*PETPrep* provides the ``-w <PATH>`` command line argument to set a customized temporary
folder (the *working directory*, in the following) for the *Nipype* workflow engine.
By default, *PETPrep* configures the *working directory* to be ``$PWD/work/``.
Therefore, if your *PETPrep* process crashes and you attempt to re-run it reusing
as much as it could from the previous run, you can either make sure that
the default ``$PWD/work/`` points to a reasonable, reusable path in your environment or
configure a better location with ``-w <PATH>``.
When rerunning, keep the same *PETPrep* version, command-line options, output
directory, and working directory unless you intentionally want to recompute the
workflow.
Before restarting, inspect the crashfiles and logs under
``<output dir>/sub-<participant_label>/log`` to identify whether the failure was
caused by a data issue, missing dependency, resource limit, or interrupted job.

How can I reuse derivatives safely across staged or session-split workflows?
-----------------------------------------------------------------------------
Use ``--derivatives`` for reusable BIDS Derivatives, and treat the derivative
dataset as part of the analysis provenance.
The raw BIDS dataset, *PETPrep* version or container build, anatomical-reference
strategy, and analysis-defining options should match the workflow that generated
the derivatives unless the difference is intentional and documented.

For staged workflows, write each stage to a stable output directory and use a
separate working directory for each run.
For example, an anatomical or minimal-derivatives stage can be generated first,
then supplied to later PET-processing runs with
``--derivatives petprep-anat=/path/to/derivatives``.
The text before ``=`` is a local nickname for that derivative source, recorded
in the output metadata as a dataset link.
It can refer to derivatives from an earlier *PETPrep* run or to another
compatible BIDS Derivatives dataset, such as sMRIPrep outputs.
If the nickname is omitted, *PETPrep* uses the directory name.
Do not have multiple jobs generate or update the same subject/session outputs in
the same derivative tree at the same time.
Shared FreeSurfer outputs may be reused with ``--fs-subjects-dir``, but they
should not be generated concurrently for the same subject.

For session-split longitudinal workflows, make sure the reused derivatives match
the anatomical strategy.
Derivatives generated from a shared subject-level anatomical reference should
not be mixed with a ``sessionwise`` workflow unless that is the intended design.
When sessions are processed independently, use
``--subject-anatomical-reference sessionwise`` for the common per-session case,
or use matching ``pet`` and ``t1w`` filters with separate output and working
directories for custom groupings.
Afterward, merge derivative datasets only as a deliberate provenance step, and
keep the original logs, configuration files, and ``dataset_description.json``
records.

Can I use *PETPrep* for longitudinal studies?
----------------------------------------------
Yes, but the preprocessing strategy should be chosen before the full dataset is
run.
PET-to-anatomical registration, segmentation, PVC, reference-region masks, and
regional TAC extraction all depend on the anatomical reference and derived
segmentations.
Changing how anatomy is handled across visits can therefore change the PET
derivatives that downstream models see.

When a subject has multiple T1w images, *PETPrep* builds a subject-level
anatomical reference.
The anatomical strategy is selected with ``--subject-anatomical-reference``:

* ``first-lex``: build one subject-level anatomical reference aligned to the
  first T1w image in lexicographic order.
  This is the default and is usually appropriate when anatomy can be treated as
  stable and the goal is to compare PET sessions in one subject space.
* ``unbiased``: build one subject-level anatomical reference with FreeSurfer's
  `mri_robust_template`_ so that the template is unbiased, or approximately
  equidistant from all selected T1w images.
  The deprecated ``--longitudinal`` flag is an alias for this mode.
* ``sessionwise``: process each selected session independently.
  PET and anatomical inputs are restricted to the same session, and each session
  gets its own anatomical reference.
  If a session contains multiple T1w images, they are combined within that
  session using the ``first-lex`` strategy.

Use ``sessionwise`` when visits should not share anatomical derivatives, for
example when age, disease progression, surgery, treatment, or scanner/protocol
changes make a shared anatomical reference questionable.
Use ``--session-label`` to limit which sessions enter the sessionwise workflow.
For custom groupings that are not one session at a time, use
``--bids-filter-file`` with matching ``pet`` and ``t1w`` filters and run each
group with separate output and working directories.
If you intentionally want to run anatomy once and reuse it later, generate the
anatomical derivatives first and provide them to subsequent PET runs with
``--derivatives``.

Keep tracer, reconstruction, PET reference, PET-to-anatomy registration,
segmentation, reference-region, PVC, and output-space choices consistent across
visits that will be compared directly.
When substantial anatomical changes are expected, special considerations must be
taken.
Some examples follow:

* Surgery: use only pre-operation sessions for the anatomical data, typically by
  omitting post-operation T1w images from the inputs used to build the
  anatomical reference.
* Developing and elderly populations: there is currently no single standard
  strategy. Process sessions independently when anatomical change is part of the
  research question, or group sessions only when that choice is justified by the
  study design.
  As `suggested by U. Tooley at NeuroStars.org
  <https://neurostars.org/t/petprep-how-to-reuse-longitudinal-and-pre-run-freesurfer/4585/15>`__,
  the anatomical fast-track can also be combined with ``--bids-filter-file`` to
  process sessions fully independently, or grouped by study-design criteria.


How to decrease *PETPrep* runtime when working with large datasets?
--------------------------------------------------------------------
*PETPrep* leverages PyBIDS to produce a layout, which indexes the input BIDS dataset and facilitates file queries.
Depending on the amount of files and metadata within the BIDS dataset, this process can be time-intensive.
The ``--bids-database-dir <database_dir>`` option can be used to pass in an
already indexed BIDS layout.

The default *PETPrep* layout can be generated by running the following shell command (requires PyBIDS 0.13.2 or greater)::

  pybids layout <bids_root> <database_dir> --no-validate --index-metadata

where ``<bids_root>`` indicates the root path of the BIDS dataset, and ``<database_dir>``
is the path where the pre-indexed layout is created - which is then passed into *PETPrep*.

By using the ``--force-index`` and ``--ignore`` options,
finer control can be achieved of what files are visible to PETPrep.

Note that any discrepancies between the pre-indexed database and
the BIDS dataset complicate the provenance of PETPrep derivatives.
If ``--bids-database-dir`` is used, the referenced directory should be
preserved for the sake of reporting and reproducibility.

For large studies, also consider running one subject per job, reusing existing
FreeSurfer outputs with ``--fs-subjects-dir`` when appropriate, and using
``--derivatives`` to reuse valid precomputed derivatives.
The ``--level`` option can reduce the number of derivatives written during early
workflow stages, but make sure the selected level still produces the outputs
needed for downstream kinetic modeling, PVC, surface analysis, or quality
control.
