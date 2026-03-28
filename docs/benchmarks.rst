.. include:: links.rst

----------------------
Performance benchmarks
----------------------

This page defines a reproducible benchmark experiment for PETPrep using PET-BIDS datasets.
The structure mirrors the historical fMRIPrep-style benchmark layout (datasets, command matrix,
machine details, and runtime/storage outcomes), but all inputs and options are PET-focused.

Datasets and commands
---------------------

Datasets
~~~~~~~~

+------------+---------------------------------------------------------------------------------------------------------------+
| Dataset    | Description                                                                                                   |
+============+===============================================================================================================+
| A          | 6 participants, static FDG PET (1 PET run/participant), 1 T1w (all participants), 2 T2w (subset of 2).      |
+------------+---------------------------------------------------------------------------------------------------------------+
| B          | 8 participants, dynamic PET (180-240 frames/run), 2 PET runs/participant, 1 T1w, with tracer metadata.      |
+------------+---------------------------------------------------------------------------------------------------------------+

PETPrep versions and modes
~~~~~~~~~~~~~~~~~~~~~~~~~~

All commands take the form ``petprep sourcedata/raw . participant $OPTIONS``.
The option matrix below benchmarks the same processing-level split used by PETPrep
(``minimal`` vs ``full``), plus an intermediate ``resampling`` level.

+----------------------+------------------------------------------------------------------------------------------------+
| Version / Mode       | Options                                                                                        |
+======================+================================================================================================+
| 0.0.4 (full)         | ``--level full --output-spaces MNI152NLin2009cAsym``                                          |
+----------------------+------------------------------------------------------------------------------------------------+
| 0.0.5 (minimal)      | ``--level minimal --output-spaces MNI152NLin2009cAsym``                                       |
+----------------------+------------------------------------------------------------------------------------------------+
| 0.0.5 (resampling)   | ``--level resampling --output-spaces MNI152NLin2009cAsym``                                    |
+----------------------+------------------------------------------------------------------------------------------------+
| 0.0.5 (full + PVC)   | ``--level full --output-spaces MNI152NLin2009cAsym --pvc-tool petsurfer --pvc-method gtm``   |
|                      | ``--pvc-psf 6 6 6``                                                                            |
+----------------------+------------------------------------------------------------------------------------------------+

Machine details
~~~~~~~~~~~~~~~

Run each benchmark on an otherwise idle system and report:

* Processor model and logical core count
* RAM (GiB)
* Storage type/capacity (NVMe/SATA, etc.)
* Operating system (distribution + version)
* Container/runtime used (for example Docker image tag)

Recommended environment for reproducibility:

* PETPrep container image from Docker Hub: ``nipreps/petprep:<tag>``
* Inputs mounted read-only, outputs/work mounted read-write
* No concurrent heavy jobs

Reproducible execution template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example command pattern (replace paths and ``$OPTIONS``):

.. code-block:: bash

   docker run --rm \
     -v $PWD/sourcedata:/data:ro \
     -v $PWD/derivatives:/out \
     -v $PWD/work:/work \
     nipreps/petprep:<tag> \
     /data/raw /out participant \
     -w /work $OPTIONS

For each run, collect:

* Wall-clock runtime (start/end timestamps)
* Peak scratch size and file count (``work`` directory)
* Final derivatives size and file count (output directory)

Suggested collection commands:

.. code-block:: bash

   /usr/bin/time -v <petprep-command>
   du -sh work out
   find work -type f | wc -l
   find out -type f | wc -l

Benchmarks
----------

Dataset A
~~~~~~~~~

+----------------------+----------+--------------+---------------+-------------+--------------+
| Version / Mode       | Runtime  | Scratch Size | Scratch Files | Output Size | Output Files |
+======================+==========+==============+===============+=============+==============+
| 0.0.4 (full)         | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+
| 0.0.5 (minimal)      | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+
| 0.0.5 (resampling)   | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+
| 0.0.5 (full + PVC)   | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+

Dataset B
~~~~~~~~~

+----------------------+----------+--------------+---------------+-------------+--------------+
| Version / Mode       | Runtime  | Scratch Size | Scratch Files | Output Size | Output Files |
+======================+==========+==============+===============+=============+==============+
| 0.0.4 (full)         | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+
| 0.0.5 (minimal)      | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+
| 0.0.5 (resampling)   | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+
| 0.0.5 (full + PVC)   | TBD      | TBD          | TBD           | TBD         | TBD          |
+----------------------+----------+--------------+---------------+-------------+--------------+
