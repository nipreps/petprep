PET-only processing (experimental)
==================================

``--pet-only`` builds a PET preprocessing workflow without a participant T1w
image. The standard spaces requested with ``--output-spaces`` are both the
registration targets and the spaces of the quantitative PET and atlas outputs.
The default is ``MNI152NLin2009cAsym:res-1``, with no subject T1w output.
When ``--output-spaces`` is omitted, PET, atlas labels, sampling support, TACs,
and morphometry therefore use the template's 1 mm isotropic grid. Explicit
output-space selections remain supported: ``:res-2`` uses the 2 mm template
grid, and ``:res-native`` retains PET voxel spacing in template space.
Each requested space is processed independently after shared motion estimation.

This first implementation uses the **T1w template image** in the requested
TemplateFlow space as the fixed registration reference. This does not require
a subject MRI. Tracer-specific PET template discovery and selection are future
extensions to the resource resolver; no tracer is automatically matched today.
Direct PET-to-T1w-template registration remains experimental and needs
evaluation for the intended tracer, uptake window, and population.

Example
-------

.. code-block:: console

   petprep /bids /out participant \
     --pet-only \
     --output-spaces MNI152NLin2009cAsym:res-1 \
     --seg Schaefer2018100Parcels7Networks \
     --petref first5min

The example selects a cortical atlas already represented in PETPrep's atlas
configuration. It illustrates the interface, not a validated registration
recipe for every tracer. Required template and atlas files must be accessible
through TemplateFlow, including its local cache for offline processing.

Motion, reference, and normalization
-----------------------------------

For dynamic PET, PETPrep's existing HMC workflow estimates per-frame transforms.
The registration reference is then built from motion-corrected PET. For static
3D or singleton 4D input, no HMC estimation is needed: identity motion transforms
and a 3D reference are used.

``--petref`` continues to control reference construction. PET-only processing
defaults to ``twa`` and supports ``template``, ``twa``, ``sum``, and ``first5min``.
``template`` refers to the within-run HMC template, independently of the fixed
TemplateFlow target. If motion correction is explicitly disabled, this strategy
uses a time-weighted reference instead. ``auto`` currently depends on anatomical
registration scoring and is rejected when explicitly requested in PET-only mode.
``first5min`` uses the 0–300 second interval in ``FrameTimesStart`` coordinates;
it does not silently reinterpret the window as injection-relative time.

Motion estimation retains PETPrep's existing settings and external dependencies,
including FreeSurfer's ``MRIConvert`` and ``RobustTemplate`` for dynamic runs.
Frames preceding the HMC start threshold inherit the first estimated transform;
consider this behavior when early frames define the normalization reference.

Stages and derivative reuse
---------------------------

The PET-only workflow separates and logs five stages:

1. **Motion estimation or reuse** (``pet_motion_wf``). Discover precomputed
   motion derivatives through ``--derivatives``, as in the anatomical workflow.
   If a matching pair is available, omit the HMC estimation workflow. Otherwise,
   estimate motion using the existing ``init_pet_hmc_wf``. Static PET and
   ``--hmc-off`` use identity transforms.
2. **Reference construction** (``pet_reference_wf``). Apply the selected motion
   transforms and construct the requested ``--petref`` image. ``template`` uses
   the HMC reference directly. Changing ``--petref`` does not require repeating
   motion estimation. An existing reference with the requested strategy label
   and matching sidecar settings skips reference construction.
3. **Direct template registration** (``pet_template_reg_wf``). Register the
   selected PET reference to each requested TemplateFlow target, or reuse a
   transform whose sidecar identifies the same reference and target images.
4. **Resampling** (``pet_volumetric_resample_wf`` and ``support_resample_wf``).
   Apply motion and template transforms to the original frames and sampling
   support in one interpolation, or reuse the normalized PET and support images.
5. **Atlas measurements** (``template_segmentation`` and ``pet_tacs_wf``).
   Prepare template-space labels and produce morphometry and TAC derivatives.

Previously estimated motion from an anatomical or PET-only run, and reference,
registration, and resampling results from a PET-only run, can be reused with a
fresh or cleaned working directory:

.. code-block:: console

   petprep /bids /out participant \
     --pet-only --seg MASSP20 --petref first5min \
     --derivatives petprep=/previous/petprep \
     -w /work --clean-workdir

Motion reuse requires both native ``desc-hmc_petref.nii.gz`` and
``from-orig_to-petref_mode-image[_desc-hmc]_xfm.txt`` for the same acquisition
in the same derivative dataset. Historical transform filenames without
``desc-hmc`` are supported. New PET-only runs save this pair using the existing
PETPrep derivative writers, separately from ``desc-<petref>_petref``.
The selected pair is checked for a 3D reference and the correct number of motion
transforms. Incomplete or ambiguous matches raise an error rather than combining
unrelated references and transforms. A later unrelated or incomplete derivative
dataset does not replace an already discovered complete pair.

With reusable motion derivatives, the log reports
``PET Stage 1: Found head motion correction transforms and petref - skipping Stage 1``.
When no matching registration reference exists, the reference stage resamples frames
to construct ``first5min``, ``twa``, or ``sum``; this applies existing motion
transforms without estimating motion again. ``--hmc-off`` ignores cached motion
derivatives. To estimate motion with changed HMC settings, omit the corresponding
derivative dataset.

Stages 2–4 use derivative filenames and their ordinary JSON sidecars:

* The native reference is labelled by ``--petref``, for example
  ``sub-001_ses-01_desc-first5min_petref.nii.gz``. PETPrep checks for this file
  for the same acquisition. Its sidecar records the reference strategy, frame
  timing, and whether motion correction was enabled.
* Transform sidecars record ``Sources``: the strategy-labelled PET reference,
  the TemplateFlow target image, and its brain mask. PETPrep checks these names,
  the reference settings, target space/cohort/resolution, and the ANTs settings
  (including ``--sloppy``). ``RegistrationMask`` records the template mask source,
  physical dilation radius, and masked stages. Both transforms and the warped
  reference must exist and have matching mask settings.
* Normalized PET and support sidecars identify the source PET and transform in
  ``Sources`` and retain the transform's ``RegistrationSources`` and settings.
  Both images must match before resampling can be skipped.

There are no separate cache manifests or content hashes for stage reuse.
Matching uses the derivative dataset passed to ``--derivatives``, independently
of the working directory, and is evaluated separately for each target.
The 1 mm default writes ``res-1`` derivatives. Earlier ``res-native`` or
``res-2`` target derivatives do not satisfy this request; compatible native
motion and registration-reference derivatives remain reusable.

.. list-table:: Effects of changing settings
   :header-rows: 1
   :widths: 35 65

   * - Change
     - Processing
   * - Only ``--seg``
     - Reuse stages 2–4; resolve the new atlas and regenerate morphometry and TACs.
   * - ``--petref`` or frame timing
     - Reuse a matching labelled reference if available; otherwise build it.
       Refit registration and resample when their provenance no longer matches.
       Retain reusable motion.
   * - Target space, cohort, resolution, template filename, or ANTs recipe
     - Reuse the native reference; rerun registration and resampling for affected targets.
   * - Missing normalized PET or support image
     - Reuse the reference and registration; regenerate resampled outputs.

Segmentation is excluded from the reuse checks. Atlas resource
provenance, overlays, morphometry, and TACs are generated for the current request.
Anatomical PET-to-T1w transforms are not used for direct template registration.

Older ``desc-registration_petref`` files do not identify a reference strategy,
and transforms without source provenance cannot establish a match. Their motion
derivatives remain reusable; stages 2–4 run once to write the labelled references
and sidecars. Native references for different strategies coexist. Transforms
and normalized PET represent the latest settings for each acquisition and target.
This filename-based approach assumes input images and motion derivatives are
unchanged in place. If their contents are replaced under the same names, remove
the affected downstream derivatives before reuse.

Template registration
---------------------

The new ``init_pet_template_reg_wf`` follows CATNIP's registration strategy:

* Header, intensity center-of-mass, and principal-axis initializations; an
  indeterminate principal-axis candidate is omitted.
* ANTs rigid, affine, then SyN stages with mutual information, a four-level
  pyramid, winsorization, and no histogram matching.
* Candidate comparison on the fixed template brain mask, with the highest MI
  selected. Forward and inverse composite transforms and candidate scores are
  saved. Initialization and candidate comparison use the original brain mask.
* Only the SyN metric is restricted to the template brain mask dilated by 5 mm.
  Dilation uses a Euclidean sphere in physical space, accounting for the template
  voxel geometry. The extra margin retains background around the cortical
  boundary. Rigid and affine registration remain unmasked.

This draft fails if an ANTs candidate fails. CATNIP's candidate recovery,
affine plausibility gates, and more extensive QC have not yet been ported.
The registration overlay and candidate scores support inspection; similarity
scores alone do not establish anatomical accuracy.

The PET-only default uses more conservative nonlinear settings:

* ``SyN[0.1,3,0.5]`` adds total displacement-field smoothing (previously
  ``SyN[0.1,3,0]``). The third parameter is an ANTs variance in voxel units.
* SyN image smoothing is ``3x2x1x0mm`` (previously ``3x2x1x0vox``).
* SyN iterations are ``200x100x50x0`` (previously ``200x100x50x25``), with
  shrink factors ``8x4x2x1``. Optimization stops at shrink factor 2; the final
  zero-iteration level represents the deformation on the full fixed-image grid
  for downstream resampling. ``--sloppy`` uses ``20x10x5x0`` iterations.

Rigid and affine settings and candidate selection retain the original recipe.
The finest optimized nonlinear scale is twice the fixed template voxel spacing
(2 mm for a 1 mm template). This does not change the output grid or smooth the
quantitative PET derivatives. These defaults are intended for comparison;
improved anatomical alignment still needs evaluation on representative data.

The existing sidecar checks detect changed ANTs or registration-mask settings and
rerun registration and resampling while retaining compatible motion and reference
derivatives. Older transforms without ``RegistrationMask`` metadata are refitted.
Subsequent runs with the same recipe reuse these stages as before. To compare
with earlier results, use a separate output directory and pass the previous
derivative datasets through ``--derivatives``, keeping ``--petref`` and
``--output-spaces`` the same. No additional registration flags are needed.

Motion and direct normalization transforms are composed when resampling the
original validated PET frames. Each template-space PET output therefore uses
one spatial interpolation of the quantitative frames. Reference construction
uses its own motion-corrected intermediate. Registration intensity processing
is not applied to quantitative PET output, and no Jacobian modulation is used.

Segmentation and TACs
---------------------

The same ``--seg`` names are accepted in the anatomical and PET-only workflows.
The processing mode determines how the segmentation is obtained:

.. list-table:: Segmentation sources
   :header-rows: 1
   :widths: 30 35 35

   * - ``--seg`` choice
     - Anatomical workflow
     - ``--pet-only`` workflow
   * - ``gtm``, ``brainstem``, ``thalamicNuclei``, ``hippocampusAmygdala``,
       ``wm``, ``aparcaseg``, ``raphe``, ``limbic``
     - Estimate from subject anatomy using the existing segmentation method.
     - Fetch the corresponding TemplateFlow atlas image and label table.
   * - ``HOCPA``, ``Schaefer2018*``, ``MASSP20``
     - Fetch the configured atlas and transform it to anatomical space.
     - Fetch the atlas image in each requested output space and its label table.

PET-only processing requires **both the image and label table from TemplateFlow**.
It does not run subject-specific segmentation tools or substitute package/file
resources. The local TemplateFlow cache remains supported. Accepting a ``--seg``
name does not imply that its resources have already been published for every
template.

Existing configured atlases retain their TemplateFlow atlas/description
selectors. The atlas image must be in the requested output space; PETPrep does
not substitute an image from another template or add that template to the output
list. An explicit resource-level ``template`` in the atlas configuration can
identify a shared **label table**, since an index/name table has no spatial grid.
For HOCPA, the image comes from the requested template and the shared table is
``tpl-MNI152NLin6Asym_atlas-HOCPA_dseg.tsv`` in TemplateFlow. This handles the
HOCPA image in ``MNI152NLin2009cAsym``, which has no accompanying table in that
template's collection. Every image label must be represented in the table.
The resource provenance records the actual template query and hash separately
for the atlas image and label table.

For names such as ``gtm`` and ``brainstem``, the draft queries the requested
template for ``atlas-<name>`` discrete ``dseg`` images and ``dseg.tsv`` label
tables. These files must actually be published or installed in the local
TemplateFlow cache. Naming here is a proposed resource contract; the software
does not create or claim availability of those atlas resources. Missing or
ambiguous matches fail with the requested space and segmentation identified.
Label tables require unique ``index`` and ``name`` columns.

The atlas image is sampled onto the output PET grid with nearest-neighbor
interpolation. PETPrep's existing TAC calculation and atlas morphometry helper
are reused. There is no subject cortical-ribbon mask in this branch.

TACs contain one row per frame, including the original frame start/end times.
The implementation tracks image support separately from activity, so a true
zero concentration remains a valid observation. A region with any voxel outside
measured support in a frame receives ``n/a`` for that frame; absent atlas regions
also retain their columns as ``n/a``. The support image and coverage policy are
saved for review. Quantitative units and resolved, inherited BIDS metadata are
retained in the output sidecars.

``morph.tsv`` reports **template-region volumes on the output grid**. These are
not estimates of individual anatomical volume or atrophy. Its sidecar records
``SubjectSpecific: false`` and the measurement space. Voxel-grid differences,
including ``res-native``, can change these discretized atlas volumes.

Outputs and current scope
-------------------------

The draft writes, for each target space:

* Quantitative ``desc-preproc_pet.nii.gz`` suitable for subsequent kinetic
  modeling in template space.
* Atlas ``seg-<name>_dseg.nii.gz``, the label table, ``morph.tsv``, and
  ``desc-preproc_seg-<name>_tacs.tsv`` with metadata.
* Forward/inverse PET-reference-to-template composite transforms, registration
  candidate records, resource queries and hashes, a sampling-support image,
  and a PET/template overlay.

Acquisition entities, output space, segmentation where applicable, and numeric
resolution distinguish outputs. The PET-only writer extends NiPreps filename
patterns for PET-space atlas tables and resolution-specific outputs. The native
registration reference and per-frame HMC transforms are also retained.

Kinetic modeling itself remains a downstream step. This branch currently
requires full processing and volumetric standard output spaces. Individual
surfaces/CIFTI, PVC, reference-mask options, and reuse of precomputed anatomical
derivatives are outside this draft. The existing anatomical workflow remains
the default when ``--pet-only`` is absent.

Validation
----------

Tests cover PET-only subject selection and graph construction for 3D/4D data,
motion-derivative discovery and reuse with fresh working directories, application
of cached transforms before reference construction, incomplete/mismatched caches,
static references, template-space label/morphometry/TAC behavior, incomplete
support, exact resource queries, and initialization direction. ANTs-dependent
tests exercise real nonlinear registration and compare nonlinear-plus-motion
resampling against ``antsApplyTransforms`` with nontrivial image orientation.
A complete synthetic static PET-only run exercises derivative writing and a
second run with a different segmentation and fresh work directory. Tests verify
that reference estimation, ANTs registration, and PET/support resampling are
absent from the reused graph, and that changed settings invalidate the relevant
stages.
These tests check implementation contracts; representative PET/MRI datasets
are still needed to assess registration accuracy and regional TAC bias.
