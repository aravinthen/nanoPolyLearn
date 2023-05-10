# Program Name: main.py
# Author: Aravinthen Rajkumar
# Description: This is where data is cleaned and models are learned.

import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
# import GPy
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import scipy.stats as st
import time

class Data:
    def __init__(self, path2file, points, indices):
        self.path2file = path2file
        self.points = points
        self.indices = indices
        self.stress_strain = {}

    def getStressStrain(self, remove_lt0=False):
        ststdata = {}
        dpath = self.path2file
        folders = os.listdir(dpath)
        for i in folders:
            # get the max extension to use as a key for the ststdata dictionary
            max_ext = []
            for letter in i:
                if letter.isnumeric() or letter == ".":
                    max_ext.append(letter)
            max_ext = float("".join(max_ext))

            # get data values for each of the curves
            # opens the folder for a full extension (multiple curves)
            all_stress = {}

            index = 1 # the index of the file being read
            for files in sorted(list(os.listdir(dpath+"/"+i))):        
                # opens one of the files that contains the extension data
                with open(dpath+"/"+i+"/"+files) as df:
                    data = [j.split('\t') for j in ([i.strip() for i in df][1:])]

                    # correct for data points
                    data=data[::self.points]

                    stress_strain = [] 
                    for line in data:
                        data_point = [float(line[i]) for i in self.indices]
                        data_point = [data_point[0], -data_point[1]]

                        if not remove_lt0:
                            stress_strain.append(data_point)
                        else:
                            if data_point[1] < 0:
                                stress_strain.append([data_point[0], 0.0])
                            else:
                                stress_strain.append(data_point)

                    all_stress[index] = stress_strain
                    index+=1

            ststdata[max_ext] = all_stress

        self.stress_strain = ststdata

    def shareData(self, numshare=1):
        # this function adds some of the data the end of one curve the the beginning of another.
        # it is necessary because the curves *are* connected.
        # numshare: the number of datapoints shared between each curve
        for i in self.stress_strain:
            for c in range(1, len(self.stress_strain[i])): # c stands for curve
                curve1 = self.stress_strain[i][c]
                curve2 = self.stress_strain[i][c+1]

                # add the shared points to each curve
                curve1 = curve1 + curve2[0:numshare]
                curve2 = curve1[len(curve1)-numshare:len(curve1)] + curve2 

                self.stress_strain[i][c] = curve1
                self.stress_strain[i][c+1] = curve2
    

class MultiGP:
    def __init__(self, data):
        self.data = data
        self.numcs = len(data[[i for i in data][0]])

        # this is where the gps that are trained will find themselves.    
        self.stopping = None
        self.gps = {}
        self.X = []
        self.y = []

    def curve_select(self, alldata, index):
        # alldata should be a dictionary that contains multiple dictionaries.
        # those dictionaries should have an index corresponding to a given part of the motion
        # The output of this function should be a reduced version of all data.
        # Key: unloading point
        # Data: all the data associated with the relevant index

        reduced = {}
        for unloading in alldata:
            for i in alldata[unloading]:
                if i == index:
                    reduced[unloading] = alldata[unloading][i]

        return reduced        
        
    def trainStopping(self, ):
        X = []
        y = []
        for i in range(0, self.numcs, 2):
            dataset = self.curve_select(self.data, i)
            for curve in dataset:
                X.append([curve, i])
                y.append(dataset[curve][-1][0])

        X = np.stack(np.array(X))
        y = np.array(y)
        
        kernel = RBF([5,5], (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e-1))
        
        self.stopping = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
        self.stopping.fit(X, y)
        print("Stopping distances fitted.")

    def trainCurves(self,):
        label = 1
        for i in range(self.numcs):
            dataset = self.curve_select(self.data, i+1)

            X = []
            y = []

            x1 = []
            x2 = []

            for curve in dataset:
                for point in dataset[curve]:
                    X.append([curve, point[0]])
                    y.append(point[1])
                    x1.append(point[0])
                    x2.append(point[1])

            X = np.stack(np.array(X))
            y = np.array(y)

            kernel = RBF([5,5],
                         (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
            
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)            
            gp.fit(X, y)
            print(f"Curve {i} fitted.")

            self.gps[label] = gp
            label += 1

        print("All curve models trained!")        
