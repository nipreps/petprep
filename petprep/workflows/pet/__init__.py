# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

Pre-processing PET workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: petprep.workflows.pet.base
.. automodule:: petprep.workflows.pet.hmc
.. automodule:: petprep.workflows.pet.stc
.. automodule:: petprep.workflows.pet.registration
.. automodule:: petprep.workflows.pet.resampling
.. automodule:: petprep.workflows.pet.confounds


"""

from .confounds import init_pet_confs_wf
from .hmc import init_pet_hmc_wf
from .registration import init_pet_reg_wf
from .resampling import init_pet_surf_wf

__all__ = [
    'init_pet_confs_wf',
    'init_pet_hmc_wf',
    'init_pet_reg_wf',
    'init_pet_surf_wf',
]
