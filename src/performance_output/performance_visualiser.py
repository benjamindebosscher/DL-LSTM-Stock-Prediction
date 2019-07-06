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
#data = data.loc[data['Data size']==5000]
#data = data.loc[data['n_predict_once']==50]

X_all = data[setting] # All values of n_predict_one
Y_all = data['correct'] # All correctness parameters
X_dat = list(set(X_all))
X_dat = sorted(X_dat)
Y_dat = []
plt.figure()
ct = 0
for X in X_dat:
	data_slice = data.loc[data[setting]==X]
	lab = setting+'= '+ str(X)
	lab = 'Prediction length = '+ str(X)
	print('Pred ', X, ' average: ', np.average(data_slice['correct']), ' st dev', np.std(data_slice['correct']))
	plt.scatter(range(ct, ct+len(data_slice)), data_slice['correct'], label = lab)
	ct += len(data_slice)
plt.legend(prop={'size': 15})
plt.tick_params(labelsize=15)
plt.ylabel('Up/down Correctness [%]', size = '20')
plt.xlabel('Run number', size = 20)

plt.title('Effects of varying '+str(setting), size = '30')
plt.title('Varying Prediction Length', size = '30')
plt.show()