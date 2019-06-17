'''AEX stock predication algorithm based on a LSTM
        Authors:    Benjamin De Bosscher
                    Jeff Maes
                    Daan Cuppen
                    Isaac Seminck
                    Kipras Paliušis
        Based on a tutorial of Thushan Ganegedara (https://www.datacamp.com/community/tutorials/lstm-python-stock-market)
'''

from src.data_operations.import_as_dict_backup import get_data
from src.data_operations.preprocessing_backup import PreProc
from src.LSTM_backup import LSTM
from src.makeplots_backup import prediction

# =============================================================================
# Preprocessing
# =============================================================================

# Import data
data_source = 'git'
market = 'AEX'
stocks = get_data(data_source, market)

# ONLY FOR NOW, SHOULD BE CHANGED!!
df = stocks['URW']

# Preprocessing data
split_datapoint = 5000         #5000 for PHIA
smoothing_window_size = 1000    #1000 for PHIA

pp_data = PreProc(df)
pp_data.splitdata(split_datapoint)
pp_data.normalize_smooth(smoothing_window_size, EMA=0.0, gamma=0.1)

# =============================================================================
# Define and apply LSTM
# =============================================================================

# Define hyperparameters
D = 1                           # Dimensionality of the data. Since our data is 1-D this would be 1
num_unrollings = 50             # Number of time steps you look into the future.
batch_size = 500                # Number of samples in a batch
num_nodes = [200, 200, 150]     # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes)       # number of layers
dropout = 0.2                   # Dropout amount

n_predict_once = 50

# Run LSTM
x_axis_seq, predictions_over_time, KPI = LSTM(pp_data, D, num_unrollings, batch_size, num_nodes, n_layers, dropout)

# =============================================================================
# Visualisation of the results
# =============================================================================

# Visualisation
best_prediction_epoch = 22    # Replace this with the epoch that you got the best results when running the plotting code
prediction(df, pp_data, x_axis_seq, predictions_over_time, best_prediction_epoch)
