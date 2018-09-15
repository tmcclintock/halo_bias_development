"""
In this analysis, I will fit individual snapshots using all combinations of the default Tinker parameters. This will hopefully let us find out what the optimal combination is.
"""

import numpy as np
import aemulus_extras as ae
import Aemulus_data as AD

sfs = AD.scale_factors()
zs = 1./sfs - 1
