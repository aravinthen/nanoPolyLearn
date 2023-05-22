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
            100,
            [2,5])

glue = 50
data.getStressStrain(glue=glue)
data.shareData(numshare=2)

# data.viewc(3)
# data.viewall()

# build machine learning model
multi = MultiGP(data.stress_strain)

multi.trainStopping(glue)
num_procs = 3 # multi.numc
multi.trainCurves(curve_range=num_procs)

# Use machine learning models for plotting
t0 = time.time()

numps = 5
extensions = np.array([i for i in data.stress_strain])
min_ext = extensions.min() # minimum extension
max_ext = extensions.max() # maximum ^

ext_range = np.linspace(min_ext, max_ext, numps)

load_spacing = 50
unload_spacing = 30

for extension in ext_range:
    # the very first strain value.        
    initial = multi.data[[i for i in multi.data][0]][1][0][0]
    strain = []
    stress = []
    error = []
    for procnum in range(1, num_procs+1):            
        # store the stress, strain and error curves 
        print(extension, procnum)
        
        if procnum%2 == 0: # this is a unloading curve
            # first generate predictions for the the stopping distance
            point = [[extension, procnum]]
            stop_means, stop_covs = multi.stopping.predict(point, return_cov=True)
            stop_means = stop_means[0]

            # generate trial stress positions with respect to the trained model
            spoints = np.array([[extension, s] for s in stop_means])
            stop_stress, stop_var = multi.gps[procnum].predict(spoints, return_cov=True)

            # find the minimum index
            abs_stress = [abs(i) for i in list(stop_stress)]
            stop_ind = abs_stress.index(min(abs_stress))
                               
            # then predict the stress-strain points
            strains = np.linspace(extension, list(stop_means)[stop_ind], unload_spacing)
            def_points = np.array([[extension, s] for s in strains]) # s is strain
            stress_mean, stress_cov = multi.gps[procnum].predict(def_points, return_cov=True)

            # add to global list
            stress += list(stress_mean)
            strain += list(strains)
            error += list(np.sqrt(np.diag(stress_cov)))            
            initial = strain[-1]

        else: # this is a loading curve.
            strains = np.linspace(initial, extension, load_spacing)
            if procnum == 1:
                points = np.array([[data.max_ext, s] for s in strains]) # s is strainb
            else:
                points = np.array([[extension, s] for s in strains]) # s is strain
                
            stress_mean, stress_cov = multi.gps[procnum].predict(points, return_cov=True)

            stress += list(stress_mean)
            strain += list(strains)
            error += list(np.sqrt(np.diag(stress_cov)))

        plt.plot(strain, stress)
        plt.savefig(f"extension_{extension}_{procnum}.png")


t1 = time.time()

print(f"Program complete! Time taken: {t1-t0}")
