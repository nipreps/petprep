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
