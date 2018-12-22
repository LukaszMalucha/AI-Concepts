# -*- coding: utf-8 -*-


################################################### Import Libraries & .py files

import os
import numpy as np
import random as rn
import environment
import brain
import deep_q_learning

### Reproducibility seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)


######################################################### Setting the Parameters

