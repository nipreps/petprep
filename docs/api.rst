.. include:: links.rst

================
Developers - API
================
The *NiPreps* community and contributing guidelines
---------------------------------------------------
*PETPrep* is a *NiPreps* application, and abides by the
`NiPreps Community guidelines <https://www.nipreps.org/community/>`__.
Please, make sure you have read and understood all the documentation
provided in the `NiPreps portal <https://www.nipreps.org>`__ before
you get started.

Setting up your development environment
---------------------------------------
We believe that *PETPrep* must be free to use, inspect, and critique.
Correspondingly, you should be free to modify our software to improve it
or adapt it to new use cases and we especially welcome contributions to
improve it or its documentation.

We actively direct efforts into making the scrutiny and improvement processes
as easy as possible.
As part of such efforts, we maintain some
`tips and guidelines for developers <https://www.nipreps.org/devs/devenv/>`__
to help minimize your burden if you want to modify the software.

Internal configuration system
-----------------------------

.. automodule:: petprep.config
   :members: from_dict, load, get, dumps, to_filename, init_spaces

Workflows
---------

.. automodule:: petprep.workflows.base
.. automodule:: petprep.workflows.pet
.. automodule:: petprep.workflows.pet.fit
.. automodule:: petprep.workflows.pet.apply
.. automodule:: petprep.workflows.pet.confounds
.. automodule:: petprep.workflows.pet.hmc
.. automodule:: petprep.workflows.pet.outputs
.. automodule:: petprep.workflows.pet.pvc
.. automodule:: petprep.workflows.pet.reference
.. automodule:: petprep.workflows.pet.registration
.. automodule:: petprep.workflows.pet.resampling
.. automodule:: petprep.workflows.pet.segmentation
.. automodule:: petprep.workflows.pet.tacs
