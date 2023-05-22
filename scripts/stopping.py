# Program Name: stopping.py
# Author: Aravinthen Rajkumar
# Description: playing about with the stopping function idea.

# Program Name: 
# Author: Aravinthen Rajkumar
# Description:

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../lib/')

from main import Data
from main import MultiGP

# prepare data
data = Data("/home/u1980907/Documents/Academia/Research/datasets/ml_test_res",
            300,
            [2,5])

data.getStressStrain(glue=500)
data.shareData(numshare=2)

# build machine learning model
multi = MultiGP(data.stress_strain)

multi.trainStopping(100)
