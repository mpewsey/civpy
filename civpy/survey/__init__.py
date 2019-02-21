"""
============================
Survey (:mod:`civpy.survey`)
============================

Contains functions for modeling alignments and performing spatial queries.


Spatial Models
==============
.. autosummary::
    :toctree: generated/

    SpatialHash
    SurveyPoint
    TIN

.. plot:: ../examples/survey/spatial_hash_ex2.py


Alignment
=========
.. autosummary::
    :toctree: generated/

    PI
    SurveyStake
    Alignment

.. plot:: ../examples/survey/alignment_ex1.py
"""

from .alignment import *
from .pi import *
from .spatial_hash import *
from .survey_point import *
from .survey_stake import *
from .tin import *
