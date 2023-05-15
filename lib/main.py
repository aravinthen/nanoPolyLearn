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
        self.max_ext = None

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

        # preprocessing.
        # there is no need to use the first loading curve for every extension.
        # the final (maximum) extension is more than enough.
        self.max_ext = max([i for i in ststdata])
        for i in ststdata:
            if i != self.max_ext:
                ststdata[i][1] = [] # empty the list but don't remove it

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
        
    def trainStopping(self, count):
        X = []
        y = []
        for i in range(0, self.numcs, 2):
            dataset = self.curve_select(self.data, i)
            for curve in dataset:
                points = dataset[curve]
                avg_points = sum([points[p][0]
                                  for p in range(len(points)-count, len(points))])/count
                print(avg_points)
                X.append([curve, i])
                y.append(avg_points)

        X = np.stack(np.array(X))
        y = np.array(y)
        
        kernel = RBF([5,5], (1e-2, 1e2)) # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e-1))
        
        self.stopping = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
        self.stopping.fit(X, y)
        print("Stopping distances fitted.")

    def trainCurves(self, curve_range=None):
        if curve_range == None:
            curve_range = self.numcs
            
        for i in range(1, curve_range+1):
            dataset = self.curve_select(self.data, i)

            X = []
            y = []
            for curve in dataset:
                for point in dataset[curve]:
                    X.append([curve, point[0]])
                    y.append(point[1])

            X = np.stack(np.array(X))
            y = np.array(y)

            kernel = RBF([5,5],
                         (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
            
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)            
            gp.fit(X, y)
            
            print(f"Curve {i} fitted.")
            self.gps[i] = gp

        print("All curve models trained!")        
