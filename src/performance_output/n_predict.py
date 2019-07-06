# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:19:46 2019

@author: Kiprot
this is used to plot the results of the performance of the model from the 	
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#setting = 'Num_nodes' 
#setting = 'Batch_size' 
setting = 'n_predict_once' 
data = pd.read_csv('PerformanceFiles/All_results_hyper.csv')
data = data.loc[data['Data size']==5000]
#data = data.loc[data['n_predict_once']==50]

X_all = data[setting] # All values of n_predict_one
Y_all = data['correct'] # All correctness parameters
X_dat = list(set(X_all))
X_dat = sorted(X_dat)
Y_dat = []
plt.figure()
ct = 0
y_plot = []
x_plot = []
for X in X_dat:
	data_slice = data.loc[data[setting]==X]
	x_plot.append(X)
	y_plot.append(np.max(data_slice['correct']))
plt.scatter(x_plot, y_plot)
plt.legend()
plt.ylabel('Up/down Correctness [%]', size = 'xx-large')
plt.xlabel('Prediction Length', size = 'xx-large')
plt.title('Effects of varying '+str(setting), size = 'xx-large')
plt.show()