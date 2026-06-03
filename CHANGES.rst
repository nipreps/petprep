0.0.7 (June 03, 2026)
=====================
Bug-fix release in the 0.0.x series.

  * ENH: Add uncropped anatomical fallback for PET-to-anatomical registration (#304)
  * FIX: Disable submillimeter FreeSurfer recon by default for PET workflows (#321)
  *  (#318)
  * MAINT: Stabilize HMC-off workflow test (#317)
  * ENH: Update longitudinal anatomical reference processing (#301)
  * ENH: Add aparc+aseg segmentation and corpus callosum reference mask (#314)
  * FIX: Update PETPrep outputs docs to match PET workflow derivatives (#310)
  * FIX: Align workflow documentation with PETPrep processing pipeline (#309)
  * FIX: Clarify usage docs and align PET filtering options (#308)
  * FIX: Update output-spaces documentation (#307)
  * FIX: Collapse --petref auto to template for 3D PET inputs (#306)
  * ENH: Use session-aware sMRIPrep derivative collection (#302)


0.0.6 (May 05, 2026)
====================
Bug-fix release in the 0.0.x series.

  * ENH: dynamically update boilerplate text (#297)
  * ENH: add reconstruction filtering support with --rec-label (#292)
  * ENH: add support skipping subjects without T1w or PET data (#294)
  * ENH: align colorbars in motion visualization (#296)
  * ENH: fix session labeling filtering when sessions.tsv file is present (#290)
  * FIX: removed functional ref to reports (#278)
  * ENH: Fix reference mask resampling (#285)
  * ENH: Add validation for segmentation and reference region (#284)
  * ENH: Fix HMC report FoV cropping for corrected frames (#283)
  * FIX: add perl to docker base (#276)
  * WIP: Remove fMRIPrep references (#260)
  * ENH: install MCR in dockerfile for certain segmentation workflows (#270)
  * add cleanup script for docker (#274)
  * ENH: Add PVC metadata to TAC sidecar outputs (#273)
  * MAINT: fix pytest scipy dependency issue (#259)
  * ENH: Add Schaefer 2018 atlas variants (7/17 networks, 100–1000 parcels) (#254)
  * BUG: Fix subject ID conflicts in visualizations (#182)


0.0.5 (March 19, 2026)
======================
Bug-fix release in the 0.0.x series.

  * FIX: fixed references to images in atlas segmentation docs (#252)
  * ENH: Add atlas segmentation support from templateflow and visualisation in report (#232)
  * ENH: Set auto PET reference and PET-to-anat method as defaults (#251)
  * ENH: Bump templateflow to 25.1.2 (#248)
  * MNT: PEP 639 compliance (#148)
  * MNT: do not rerun `ruff check` after `ruff format` (#149)
  * FIX: Bump nipype requirement to 1.11.0 (#245)
  * FIX: Update according to main (#216)
  * ENH: Add template atlas segmentation support (#187)


0.0.4 (February 19, 2026)
=========================
Bug-fix release in the 0.0.x series.

* maint preparation for 0.0.4 release (#238)
* Refine motion reportlet cropping mask (#234)
* Revert "WIP: Merge original PET metadata with derived fields" (#222)
* WIP: Merge original PET metadata with derived fields (#220)
* ENH: Add option to combine PET runs before preprocessing (#213)
* ENH: Add run label filtering option (#207)
* ENH: Use robust percentile threshold for PET reference mask (#204)
* ENH: Optimize co-registration of T1w to petref, and provide more options for petref generation and registrations (#185)
* Add automatic PET-to-anatomical registration selection (#193)
* Add anatomical reference selection to reports (#200)
* ENH: add tracer-label functionality for filtering (#190)
* PR 185 parser fix (#199)
* Add automatic PET reference selection (#194)
* Fix PR 194 (#197)
* Adjust PET co-registration workflow to allow for a different anatomical reference (#196)
* ANTS coregistration implementation (#188)
* ENH: Improve co-registration visualization (#180)
* ENH: Add robust co-registration between PET and MRI (#178)
* ENH: allow motion correction to be turned off (#175)
* Update to main branch (#176)
* Update to main (#174)
* ENH: Add visualization of head motion correction (#171)
* ENH: Add session-label option (#168)
* ENH: Add framewise displacement graph to visuals (#173)
* Fix connection for RBV method in pvc.py (#166)
* ENH: update PVC workflow documentation (#162)
* Update PVC documentation (#160)
* Align branch with main (#161)
* FIX: update import of segmentation data in reference mask utils (#156)
* chore(ci): Add ref parameter to checkout action in Docker workflow (#154)


0.0.3 (October 06, 2025)
========================
Bug-fix release in the 0.0.x series.

* Fix hippocampus segmentation and labels (#153)
* FIX: fix thresholding to be a percentage (#151)
* FIX: Update TACs interface to match PET-BIDS derivatives spec (#146)
* ENH: Create morph refmask and derivatives (#143)
* ENH: Add ``--combine-runs`` option to merge PET acquisitions prior to preprocessing (#xxx)
* DOC: Fix the path to the `sample_report` folder in the output doc (#97)
* DOC: Add preliminary release to "What is new" page (#103)
* REF: Remove unused parameters from PET confound workflow initialization (#107)

0.0.2 (September 16, 2025)
==========================
Bug-fix release in the 0.0.x series.

* chore(deps): Pin latest niworkflows (#142)

0.0.1 (September 16, 2025)
==========================
* Initial release of PETPrep.
* Provides preprocessing workflows for PET imaging data.

0.0.1a0 (August 19, 2025)
=========================
* Preliminary release
