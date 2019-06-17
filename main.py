'''AEX stock predication algorithm based on a LSTM
        Authors:    Benjamin De Bosscher
                    Jeff Maes
                    Daan Cuppen
                    Isaac Seminck
                    Kipras Paliušis
        Based on a tutorial of Thushan Ganegedara (https://www.datacamp.com/community/tutorials/lstm-python-stock-market)
'''

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
market = 'AEX' # 'AEX'
stocks = get_data(data_source, market)

stock_list = ['PHIA', 'REN', 'KPN']
num_stocks = 1

for i in range(num_stocks):
    # ONLY FOR NOW, SHOULD BE CHANGED!!
    df = stocks[stock_list[i]]
    #df = df[:800]

    # fraction of training data/total data
    data_frac = 0.8

    # Preprocessing data
    #split in test and train data and round to nearest 1000
    #split_datapoint = int(data_frac * df.shape[0])  
    #split_datapoint = int(round(split_datapoint / 2000.0) * 2000.0)  # 5000 for PHIA
    split_datapoint = 5000   
        
    smoothing_window_size = 1000   # 1000 for PHIA
    # Number of data points to remove. Uncomment one option to remove the first N training data points
    remove_data = 0
    #remove_data = 1000
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
    	pp_data.limitdata(remove_data)

    # =============================================================================
    # Define and apply LSTM
    # =============================================================================

    # Define hyperparameters
    D = 2                           # Dimensionality of the data. Since our data is 1-D this would be 1
    num_unrollings = 20            # Number of time steps you look into the future. (also number of batches)
    batch_size = 500                # Number of samples in a batch
    num_nodes = [200, 200, 150]     # Number of hidden nodes in each layer of the deep LSTM stack we're using
    n_layers = len(num_nodes)       # number of layers
    dropout = 0.2                   # Dropout amount

    # Define number of days to predict for in the future
    #n_predict_once = 10
    #n_predict_once = 25
    n_predict_once = 20
    #n_predict_once = 100
    #n_predict_once = 200

    best_prediction_epoch = 0;

    # Run LSTM
    x_axis_seq, predictions_over_time, run_data, KPI = LSTM(pp_data, D, num_unrollings, batch_size, num_nodes, n_layers, dropout, n_predict_once, (i+1), best_prediction_epoch)
    best_prediction_epoch = PerformanceSaver(pp_data_price, run_data, KPI, n_predict_once, num_unrollings, batch_size)

# =============================================================================
# Saving the results and finding the best epoch
# =============================================================================
#%%
    
best_epoch_choice = 1 #0 if only MSE error on test data for all epochs is taken into account, 1 if MSE on last 5 epochs is taken into account
best_prediction_epoch = PerformanceSaver(pp_data_price, run_data, KPI, n_predict_once, num_unrollings, batch_size, best_epoch_choice)
# =============================================================================
# Visualisation of the results
# =============================================================================


plot = prediction(df, pp_data, x_axis_seq, predictions_over_time, best_prediction_epoch)
