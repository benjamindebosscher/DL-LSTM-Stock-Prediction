'''AEX stock predication algorithm based on a LSTM
        Authors:    Benjamin De Bosscher
                    Jeff Maes
                    Daan Cuppen
                    Isaac Seminck
                    Kipras Paliušis
        Based on a tutorial of Thushan Ganegedara (https://www.datacamp.com/community/tutorials/lstm-python-stock-market)
'''
import numpy as np
from src.data_operations.import_as_dict import get_data
from src.data_operations.preprocessing import PreProc
from src.LSTM import LSTM
from src.makeplots import prediction
from src.performance_output.txt_saver import PerformanceSaver

# =============================================================================
# Preprocessing
# =============================================================================

# Import data
data_source = 'git'
market = 'AEX'
stocks = get_data(data_source, market)

# ONLY FOR NOW, SHOULD BE CHANGED!!
df = stocks['PHIA']

# Preprocessing data
split_datapoint = 5000      # must be integer multiple of smoothing_window_size
smoothing_window_size = 1000 #must be integer multiple of n_predict_once/number_of_unrollings
# Number of data points to remove. Uncomment one option to remove the first N training data points
remove_data = 0
#remove_data = 2000
#remove_data = 3000
#remove_data = 4000

pp_data_price = PreProc(df, "Prices")
pp_data_price.splitdata(split_datapoint)
pp_data_price.normalize_smooth(smoothing_window_size, EMA=0.0, gamma=0.1)

pp_data_volume = PreProc(df, "Volume")
pp_data_volume.splitdata(split_datapoint)
pp_data_volume.normalize_smooth(smoothing_window_size, EMA=0.0, gamma=0.1)

pp_data = []
pp_data.insert(0, pp_data_price)
pp_data.insert(1, pp_data_volume)

if remove_data!=0: # Removing data points! Or not! This if statement will know.
	for data in pp_data:
		data.limitdata(remove_data)
#%%
# =============================================================================
# Define and apply LSTM
# =============================================================================

# Define hyperparameters
#D = 2                           # Dimensionality of the data. Since our data is 1-D this would be 1
num_unrollings = 50             # Number of time steps you look into the future. (also number of batches)
#batch_size = 500                # Number of samples in a batch
#num_nodes = [200, 200, 150]     # Number of hidden nodes in each layer of the deep LSTM stack we're using

#dropout = 0.2                   # Dropout amount
### ADJUST HYPERPARAMETERS
D = 2                           # Dimensionality of the data. Since our data is 1-D this would be 1
counter = 0
print('Starting RUNS')
for batch_size in [250, 500, 750]:
	for n_node in [100, 200, 300]:
		for dropout in [0, 0.1, 0.2]:
			if counter<0:
				print('Skipping RUN number ', counter)
				counter+=1
				continue
			else:
#				n_predict_once = 25
				
#				n_predict_once = 100
				n_predict_once = 50     #
				num_nodes = [n_node, n_node, 150]
				n_layers = len(num_nodes)       # number of layers
				print('PREDICT', n_predict_once, 'num_unrollings ', num_unrollings, ' batch_size ', batch_size, ' num_nodes ', num_nodes, ' dropout ', dropout)
				# Define number of days to predict for in the future
#					n_predict_once = 10
				
				
				# Run LSTM
				x_axis_seq, predictions_over_time, run_data, KPI,  mid_data_over_time = LSTM(pp_data, D, num_unrollings, batch_size, num_nodes, n_layers, dropout, n_predict_once)
				
				# =============================================================================
				# Saving the results and finding the best epoch
				# =============================================================================
				
				# Now automatically chooses the one with the highest 'correct' 
				best_prediction_epoch = PerformanceSaver(pp_data_price, run_data, KPI, n_predict_once, num_unrollings, batch_size, dropout, num_nodes)
				counter+=1
				print('Run number ', counter, ' successfully finished. ', counter/18*100, '% done')
# =============================================================================
# Visualisation of the results
# =============================================================================
#%%
#best_prediction_epoch = 4
plot = prediction(df, pp_data, x_axis_seq, predictions_over_time, best_prediction_epoch)



