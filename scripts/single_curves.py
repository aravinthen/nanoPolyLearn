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
            10,
            [2,5])

data.getStressStrain()
data.shareData(numshare=2)

# build machine learning model
multi = MultiGP(data.stress_strain)


multi.trainStopping()
multi.trainCurves()


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
    for num_cycle in range(1,multi.numcs+1):
        # the very first strain value.
        initial = multi.data[[i for i in multi.data][0]][1][0][0]
        # store the stress, strain and error curves 
        strain = []
        stress = []
        error = []
        for procnum in range(1,num_cycle+1):
            if procnum%2 == 0: # this is a unloading curve            
                # first predict the stopping distance
                point = [[extension, procnum]]
                stop_mean, stop_cov = multi.stopping.predict(point, return_cov=True)

                # then predict the stress-strain points
                strains = np.linspace(extension, stop_mean[0], unload_spacing)
                points = np.array([[extension, s] for s in strains]) # s is strain
                stress_mean, stress_cov = multi.gps[procnum].predict(points, return_cov=True)

                # add to global list
                stress += list(stress_mean)
                strain += list(strains)
                error += list(np.sqrt(np.diag(stress_cov)))            

                initial = strain[-1]
                print(initial)

            else: # this is a loading curve.
                # the strain_points
                strains = np.linspace(initial, extension, load_spacing)
                points = np.array([[extension, s] for s in strains]) # s is strain
                stress_mean, stress_cov = multi.gps[procnum].predict(points, return_cov=True)
                
                stress += list(stress_mean)
                strain += list(strains)
                error += list(np.sqrt(np.diag(stress_cov)))

            plt.plot(strain, stress)
            plt.savefig(f"extension_{extension}_{num_cycle}.png")


t1 = time.time()

print(f"Program complete! Time taken: {t1-t0}")
