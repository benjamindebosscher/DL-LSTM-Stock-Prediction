"""
Created on Sun May 19 16:56:05 2019

@author: Kiprot
"""
import datetime as dt
import numpy as np
import pandas as pd
def PerformanceSaver(data_obj, run_data, KPI,  n_predict_once, num_unrollings, batch_size):
	''' 
	This is used to return the lowest MSE run and a text file
	with the relevant time sequence parameters:
	1) train data sequence length (done by remove_data setting)
	2) num_unrolling (how many time steps the model looks into)
	3) batch_size (number of data samples at each time step)
	4) n_predict_once (how many steps are predicted for in the future)
	and the MSE per time step in a text file
	to keep track of how changes to the time sequence length 
	affects the performance of the model
	Save format: Run date, Best epoch, KPI that lead to the best epoch, Data Size, Num_unrolling, Batch size,
	n_predict_once, KPI from that epoch
	
	'''
	SAVE_FOLDER = 'src/performance_output/PerformanceFiles/' # Folder to save the run results
	title = SAVE_FOLDER + 'Ntot' + str(np.shape(data_obj.train_data)[0]) + '_Npred' + str(n_predict_once)+'.txt' # Title for text file
	header = np.array(('Train data size =' + str(np.shape(data_obj.train_data)[0]), 
					   'Num_unrolling = ' + str(num_unrollings),
					   'batch_size = ' + str(batch_size),
					   'n_predict_once = ' + str(n_predict_once)))
	header = np.reshape(header, (4,1))
	
	run_data = run_data[1:]  # Removing an empty placeholder in the run_data
	### 
	headers = list(KPI.keys())
	indices = [] # Index list for index of each lowest KPI from every epoch
	for header in headers:
		indices.append(KPI[header].index(min(KPI[header])))
		ind = KPI[header].index(min(KPI[header])) # Index for minimum kpi
#		print('KPI: ', header, ' INDEX ', ind, ' VALUES , ', KPI[header])
	best_ind = max(set(indices), key = indices.count) # Taking the most frequent index within indices
	# Saving best prediction to text file
	best_pred = np.array('Best prediction epoch: ' + str(best_ind + 1) + ' with MSE ' +str(KPI['mse'][ind]) + ', MRE ' +str(KPI['mre'][ind]))
	header2 = ['date', 'best epoch', 'from KPI:'] +  headers
	# Collecting the text file together
	output_file = np.vstack((header,best_pred,run_data)) 
	data_temp = {'date' : dt.datetime.ctime}
		
	# Outputting text file
	np.savetxt(title, output_file, fmt = '%s')
	
	
# =============================================================================
# 				### Saving all data in CSV file
# =============================================================================
	headers = list(KPI.keys())
	indices = [] # Index list for index of each lowest KPI from every epoch
			### Read current All_data csv file
	data_saver = pd.read_csv('src/performance_output/PerformanceFiles/All_results.csv')
	for header in headers:
		ind = KPI[header].index(min(KPI[header]))
		indices.append(ind)
	ind_chosen = max(set(indices))
	header2 = ['Date', 'Best epoch', 'From KPI:', 'Data size', 'Num_unrolling', 'Batch_size', 'n_predict_once'] +  headers
	data_temp = np.zeros(np.shape(header2)[0])

	# Making the data row
	KPI_elems = ''
#	data_obj = pp_data[0] ### REMOVE THIS LATER
	res_list = [i for i, value in enumerate(indices) if value == ind_chosen]
	for i in res_list:
		KPI_elems += headers[i] + ', '
	KPI_elems = KPI_elems[:-2]
	data_temp = {header2[0] : dt.datetime.now().ctime(), header2[1] : ind_chosen, header2[2] : KPI_elems, header2[3] : np.shape(data_obj.train_data)[0], header2[4] : num_unrollings, header2[5] : batch_size, header2[6] : n_predict_once}
	for h in headers:
		data_temp[h] = KPI[h][ind_chosen]
	data_toadd = pd.DataFrame(data_temp, index = [0])
	data_all = pd.concat([data_saver, data_toadd], ignore_index = True, sort = False)
#	data_all = pd.concat([data_saver, data_toadd], ignore_index = True)
	data_all.to_csv('src/performance_output/PerformanceFiles/All_results.csv', index = False)
	print('Best prediction epoch: ', ind_chosen, ' with KPI: ')
	for key in KPI.keys():
		print('KPI: ', key, ' =', KPI[key][ind_chosen])
	
	return ind_chosen # Return best prediction epoch