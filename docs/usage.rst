.. include:: links.rst

.. _Usage :

Usage Notes
===========
.. warning::
   As of *PETPrep* 0.0.1, the software includes a tracking system
   to report usage statistics and errors. Users can opt-out using
   the ``--notrack`` command line argument.


Execution and the BIDS format
-----------------------------
The *PETPrep* workflow takes as principal input the path of the dataset
that is to be processed.
The input dataset is required to be in valid :abbr:`BIDS (Brain Imaging Data
Structure)` format, and it must include at least one T1w structural image and
(unless disabled with a flag) a PET scan.
We highly recommend that you validate your dataset with the free, online
`BIDS Validator <https://bids-standard.github.io/bids-validator/>`_.

The exact command to run *PETPrep* depends on the Installation_ method.
The common parts of the command follow the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.
Example: ::

    petprep data/bids_root/ out/ participant -w work/

Further information about BIDS and BIDS-Apps can be found at the
`NiPreps portal <https://www.nipreps.org/apps/framework/>`__.

Command-Line Arguments
----------------------
.. argparse::
   :ref: petprep.cli.parser._build_parser
   :prog: petprep
   :nodefault:
   :nodefaultconst:


The command-line interface of the docker wrapper
------------------------------------------------

.. argparse::
   :ref: petprep_docker.__main__.get_parser
   :prog: petprep-docker
   :nodefault:
   :nodefaultconst:



Limitations and reasons not to use *PETPrep*
---------------------------------------------

1. Very narrow :abbr:`FoV (field-of-view)` images oftentimes do not contain
   enough information for standard image registration methods to work correctly.
   Also, problems may arise when extracting the brain from these data.
   PETPrep supports pre-aligned PET data, and accepting pre-computed
   derivatives such as brain masks and atlases are a target of future effort.
2. *PETPrep* may also underperform for particular populations (e.g., infants) and
   non-human brains, although appropriate templates can be provided to overcome the
   issue.
3. If you are working with blocking data, be aware that the motion correction step may not perform optimally.
4. If you really want unlimited flexibility (which is obviously a double-edged sword).
5. If you want students to suffer through implementing each step for didactic purposes,
   or to learn shell-scripting or Python along the way.
6. If you are trying to reproduce some *in-house* lab pipeline.

(Reasons 4-6 were kindly provided by S. Nastase in his
`open review <https://pubpeer.com/publications/6B3E024EAEBF2C80085FDF644C2085>`__
of our `pre-print <https://doi.org/10.1101/306951>`__).

.. _fs_license:

The FreeSurfer license
----------------------
*PETPrep* uses FreeSurfer tools, which require a license to run.

To obtain a FreeSurfer license, simply register for free at
https://surfer.nmr.mgh.harvard.edu/registration.html.

When using manually-prepared environments or singularity, FreeSurfer will search
for a license key file first using the ``$FS_LICENSE`` environment variable and then
in the default path to the license key file (``$FREESURFER_HOME/license.txt``).
If using the ``--cleanenv`` flag and ``$FS_LICENSE`` is set, use ``--fs-license-file $FS_LICENSE``
to pass the license file location to *PETPrep*.

It is possible to run the docker container pointing the image to a local path
where a valid license file is stored.
For example, if the license is stored in the ``$HOME/.licenses/freesurfer/license.txt``
file on the host system: ::

    $ docker run -ti --rm \
        -v $HOME/fullds005:/data:ro \
        -v $HOME/dockerout:/out \
        -v $HOME/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        nipreps/petprep:latest \
        /data /out/out \
        participant

Using FreeSurfer can also be enabled when using ``petprep-docker``: ::

    $ petprep-docker --fs-license-file $HOME/.licenses/freesurfer/license.txt \
        /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /home/user/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        -v /path/to_output/dir:/out nipreps/petprep:0.0.1 \
        /data /out participant
    ...

If the environment variable ``$FS_LICENSE`` is set in the host system, then
it will automatically used by ``petprep-docker``. For instance, the following
would be equivalent to the latest example: ::

    $ export FS_LICENSE=$HOME/.licenses/freesurfer/license.txt
    $ petprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /home/user/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        -v /path/to_output/dir:/out nipreps/petprep:0.0.1 \
        /data /out participant
    ...


.. _prev_derivs:

Reusing precomputed derivatives
-------------------------------

Reusing a previous, partial execution of *PETPrep*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*PETPrep* will pick up where it left off a previous execution, so long as the work directory
points to the same location, and this directory has not been changed/manipulated.
Some workflow nodes will rerun unconditionally, so there will always be some amount of
reprocessing.

Using a previous run of *FreeSurfer*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*PETPrep* will automatically reuse previous runs of *FreeSurfer* if a subject directory
named ``freesurfer/`` is found in the output directory (``<output_dir>/freesurfer``).
Reconstructions for each participant will be checked for completeness, and any missing
components will be recomputed.
You can use the ``--fs-subjects-dir`` flag to specify a different location to save
FreeSurfer outputs.
If precomputed results are found, they will be reused.

BIDS Derivatives reuse
~~~~~~~~~~~~~~~~~~~~~~
As of version 0.0.1, *PETPrep* can reuse precomputed derivatives that follow BIDS Derivatives
conventions. To provide derivatives to *PETPrep*, use the ``--derivatives`` (``-d``) flag one
or more times.

This mechanism replaces the earlier, more limited ``--anat-derivatives`` flag.

.. note::
   Derivatives reuse is considered *experimental*.

This feature has several intended use-cases:

  * To enable PETPrep to be run in a "minimal" mode, where only the most essential
    derivatives are generated. This can be useful for large datasets where disk space
    is a concern, or for users who only need a subset of the derivatives. Further
    derivatives may be generated later, or by a different tool.
  * To enable PETPrep to be integrated into a larger processing pipeline, where
    other tools may generate derivatives that PETPrep can use in place of its own
    steps.
  * To enable users to substitute their own custom derivatives for those generated
    by PETPrep. For example, a user may wish to use a different brain extraction
    tool, or a different registration tool, and then use PETPrep to generate the
    remaining derivatives.
  * To enable complicated meta-workflows, where PETPrep is run multiple times
    with different options, and the results are combined. Processing of all sessions simultaneously
    would be impractical, but the anatomical processing can be done once, and
    then the PET processing can be done separately for each session.

See also the ``--level`` flag, which can be used to control which derivatives are
generated.

Head motion correction
----------------------
*PETPrep* can correct for head motion in the PET data.
The head motion is estimated using a frame-based robust registration approach to an unbiased mean 
volume implemented in FreeSurfer's mri_robust_register (Reuter et al., 2010), combined with 
preprocessing steps using tools from FSL (Jenkinson et al., 2012). Specifically, for the estimation 
of head motion, each frame is initially smoothed with a Gaussian filter (full-width half-maximum [FWHM] of 10 mm, --hmc-fwhm 10), 
followed by thresholding at 20% of the intensity range to reduce noise and improve registration 
accuracy (removing stripe artefacts from filtered back projection reconstructions). 
Per default, the motion is estimated selectively of frames acquired after 120 seconds post-injection of the tracer (--hmc-start-time 120),
as frames before this often contain low count statistics. Frames preceding 120 seconds were corrected 
using identical transformations as derived for the first frame after 120 seconds. The robust 
registration (mri_robust_register) algorithm utilized settings optimized for PET data: intensity 
scaling was enabled, automated sensitivity detection was activated, and the Frobenius norm threshold 
for convergence was set at 0.0001, ensuring precise and consistent alignment across frames.

By default, *PETPrep* evaluates the frames acquired after
:option:`--hmc-start-time` and initializes motion correction with the
frame exhibiting the highest tracer uptake. Provide a zero-based index
with :option:`--hmc-init-frame` to override this choice. Adding
:option:`--hmc-init-frame-fix` keeps whichever frame is selected (automatic or
manual) fixed during robust template estimation to improve reproducibility.
Iterations are automatically disabled to reduce runtime when :option:`--hmc-init-frame-fix` is
used.

When motion correction is undesirable, use :option:`--hmc-off` to disable head motion
correction entirely and keep the data unmodified apart from downstream
processing steps.

Examples: ::

    $ petprep /data/bids_root /out participant --hmc-fwhm 8 --hmc-start-time 60
    $ petprep /data/bids_root /out participant --hmc-init-frame 10 --hmc-init-frame-fix
    $ petprep /data/bids_root /out participant --hmc-off


PET reference image selection
-------------------------
Use :option:`--petref` to control how the reference volume is built from the
dynamic PET series. Each strategy uses the frame timing metadata from
``FrameTimesStart`` and ``FrameDuration`` to weight volumes; missing metadata
will raise an error before preprocessing starts.

* ``template`` (default) reuses the motion-correction template, providing a
  consistent target for downstream registration. When :option:`--hmc-off`
  disables motion correction, requesting ``template`` automatically falls back
  to ``twa`` with a warning.
* ``twa`` computes a time-weighted average, which often emphasizes later frames with
  higher counts and longer durations.
* ``sum`` produces a straightforward summed image.
* ``first5min`` averages only the first 5 minutes of PET data to capture perfusion-like
  uptake.

Anatomical reference selection
------------------------------
PETPrep uses an anatomical reference when registering PET data to the structural
image. By default, the workflow relies on the preprocessed T1w image
(:option:`--anatref t1w`), but it is also possible to use the
non-uniformity corrected ``nu.mgz`` volume produced by FreeSurfer
(:option:`--anatref nu`) when extracerebral uptake is present. 
When :option:`--anatref auto` is set, PETPrep inspects
the PET-derived brain mask volume relative to the anatomical mask. If the PET
mask is substantially larger than expected (volume ratio > 1.5), the workflow
automatically switches to ``nu.mgz`` to improve co-registration robustness.

Anatomical co-registration
--------------------------
*PETPrep* aligns the PET reference volume to the T1-weighted anatomy before
deriving downstream outputs. The anatomical image is first trimmed with
FSL's ``robustfov`` to remove the shoulder/neck and masked to limit registration to brain voxels. Choose
the registration backend with :option:`--pet2anat-method`: ``mri_coreg``
(default FreeSurfer co-registration), ``robust`` (FreeSurfer
``mri_robust_register`` with an NMI cost function), or ``ants`` (ANTs rigid
registration that consumes the unmasked T1w and a separate mask). The
:option:`--pet2anat-dof` flag controls the degrees of freedom; ``robust`` and
``ants`` are limited to rigid-body alignment and therefore require
``--pet2anat-dof 6``. All modes emit paired ITK transforms for reuse in later
resampling steps.

Segmentation
----------------
*PETPrep* can segment the brain into different brain regions and extract time activity curves from these regions.
The ``--seg`` flag selects the segmentation method to use.
Available options are ``gtm`` (default) whole-brain segmentation from freesurfer, ``brainstem``, ``wm`` (white matter), ``thalamicNuclei``, ``hippocampusAmygdala``, ``raphe``, and ``limbic``.

The ``gtm`` segmentation is a whole-brain segmentation that includes the
cerebral cortex, subcortical structures, and cerebellum.

To run the segmentation with the default ``gtm`` method, use: ::

    $ petprep /data/bids_root /out participant --seg gtm 

Partial volume correction
-------------------------
*PETPrep* can optionally correct PET images for partial volume effects.
The ``--pvc-tool`` flag selects the tool to use (``petpvc`` or ``petsurfer``),
while ``--pvc-method`` chooses the specific algorithm provided by that tool.
Available ``petpvc`` methods are ``GTM``, ``LABBE``, ``RL``, ``VC``, ``RBV``,
``LABBE+RBV``, ``RBV+VC``, ``RBV+RL``, ``LABBE+RBV+VC``, ``LABBE+RBV+RL``,
``STC``, ``MTC``, ``LABBE+MTC``, ``MTC+VC``, ``MTC+RL``, ``LABBE+MTC+VC``,
``LABBE+MTC+RL``, ``IY``, ``IY+VC``, ``IY+RL``, ``MG``, ``MG+VC`` and ``MG+RL``.
``petsurfer`` provides ``GTM``, ``MG``, ``RBV`` and ``AGTM``.
``AGTM`` runs in two steps: first the motion corrected frames are averaged
to generate a reference image. Then a geometric transfer matrix is optimised
using that reference together with the point spread function. As a
consequence, decent motion correction of the input frames and a reliable PSF
estimate are prerequisites for ``AGTM`` to succeed.
Use ``--pvc-psf`` to specify the point spread function FWHM, either as a single
value or three values. When PVC is enabled, the corrected image automatically
feeds into the remainder of the workflow, and standard-space outputs are derived
from this PVC-corrected series. The corrected data are first aligned to the
T1-weighted anatomy, and only the anatomical-to-template transforms are applied
for further resampling.

For example, to run PVC using the ``petpvc`` implementation together with the ``--seg gtm`` (default) and the ``GTM``
method with a 5 mm PSF::

    $ petprep /data/bids_root /out participant \
        --pvc-tool petpvc --pvc-method GTM --pvc-psf 5

To run ``AGTM`` with ``petsurfer`` instead::

    $ petprep /data/bids_root /out participant \
        --pvc-tool petsurfer --pvc-method AGTM --pvc-psf 5

Please note that the ``petsurfer`` implementation of PVC requires the gtm segmentation ``--seg gtm``, whereas
the ``petpvc`` implementation can use any segmentation method.

.. _cli_refmask:

Reference region masks
----------------------
*PETPrep* can build masks and time activity curves for reference regions used in pharmacokinetic quantification.
Use ``--ref-mask-name`` to select a predefined region and
``--ref-mask-index`` to override the label indices.

The available masks are and do not require ``--ref-mask-index`` to be specified:
- ``cerebellum``: Cerebellar gray matter (requires the ``--seg gtm`` option).
- ``semiovale``: White matter in the centrum semiovale (requires the ``--seg wm`` option).
- ``neocortex``: Neocortical gray matter (requires the ``--seg gtm`` option).
- ``thalamus``: Thalamic gray matter (requires the ``--seg gtm`` option).

The presets are defined in ``petprep/data/reference_mask/config.json``.


When a reference mask is created, *PETPrep* also generates a TSV table
``label-<name>_desc-ref_morph.tsv`` saved under the ``anat/`` derivatives folder. This
table mirrors the segmentation morph tables and contains three columns:
``index``, ``name`` and ``volume-mm3``.

If you want to use a custom mask, you can provide it using the ``--ref-mask-name`` and ``--ref-mask-index`` options,
specifying the name and indices of your choice for a given segmentation (``--seg``). 

For example, to extract a mask of thalamus to use as a reference region, you can run: ::

    $ petprep /data/bids_root /out participant \
        --seg gtm --ref-mask-name thalamus --ref-mask-index 10 49

The indices of the regions from a given segmentation can be found in the corresponding ``/anat/sub-<participant_label>_seg-<segmentation>_morph.tsv``.

Troubleshooting
---------------
Logs and crashfiles are output into the
``<output dir>/petprep/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`Debugging Nipype Workflows <https://miykael.github.io/nipype_tutorial/notebooks/basic_debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://petprep.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/nipreps/petprep/issues.

If you have a problem or would like to ask a question about how to use *PETPrep*,
please submit a question to `NeuroStars.org <https://neurostars.org/tag/petprep>`_ with a ``petprep`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

Previous questions about *PETPrep* are available here:
https://neurostars.org/tag/petprep/

To participate in the *PETPrep* development-related discussions please use the
following mailing list: https://mail.python.org/mailman/listinfo/neuroimaging
Please add *[petprep]* to the subject line when posting on the mailing list.


.. include:: license.rst
