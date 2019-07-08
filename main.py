'''AEX stock predication algorithm based on a LSTM
        Authors:    Benjamin De Bosscher
                    Jeff Maes
                    Daan Cuppen
                    Isaac Seminck
                    Kipras Paliušis
        Based on a tutorial of Thushan Ganegedara (https://www.datacamp.com/community/tutorials/lstm-python-stock-market)
'''
import numpy as np
import pandas as pd
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

price_df_AS = pd.DataFrame() #dataframe all stocks price
price_train_data_AS = []
price_test_data_AS = []
price_all_mid_data_AS = []

volume_df_AS = pd.DataFrame() #dataframe all stocks volume
volume_train_data_AS = []
volume_test_data_AS = []
volume_all_mid_data_AS = []


df = pd.DataFrame()
for x in stocks:
    df = stocks[x]
    # Preprocessing data
    split_datapoint = round(0.8*len(stocks[x]))     # size of taining set
    smoothing_window_size = round(split_datapoint/4) #size of smoothing window for training set
    print(smoothing_window_size)
    pp_data_price = PreProc(df, "Prices")
    pp_data_price.splitdata(split_datapoint)
    pp_data_price.normalize_smooth(smoothing_window_size, EMA=0.0, gamma=0.1)
    mid_data_price = np.concatenate([pp_data_price.train_data, pp_data_price.test_data], axis=0)
    
    pp_data_volume = PreProc(df, "Volume")
    pp_data_volume.splitdata(split_datapoint)
    pp_data_volume.normalize_smooth(smoothing_window_size, EMA=0.0, gamma=0.1)
    mid_data_volume = np.concatenate([pp_data_volume.train_data, pp_data_volume.test_data], axis=0)

    price_df_AS = price_df_AS.append(pp_data_price.df,ignore_index=True)
    price_train_data_AS = np.append(price_train_data_AS,pp_data_price.train_data)
    price_test_data_AS = np.append(price_test_data_AS,pp_data_price.test_data)
    price_all_mid_data_AS = np.append(price_all_mid_data_AS,mid_data_price)
    
    volume_df_AS = volume_df_AS.append(pp_data_volume.df,ignore_index=True)
    volume_train_data_AS = np.append(volume_train_data_AS,pp_data_volume.train_data)
    volume_test_data_AS = np.append(volume_test_data_AS,pp_data_volume.test_data)
    volume_all_mid_data_AS = np.append(volume_all_mid_data_AS,mid_data_volume)




#%%
# =============================================================================
# Define and apply LSTM
# =============================================================================

# Define hyperparameters
D = 2                           # Dimensionality of the data. Since our data is 1-D this would be 1
num_unrollings = 50             # Number of time steps you look into the future. (also number of batches)
batch_size = 250                # Number of samples in a batch
num_nodes = [200, 200, 150]     # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes)       # number of layers
dropout = 0.1                   # Dropout amount


# Define number of days to predict for in the future
n_predict_once = 50
#n_predict_once = 25
#n_predict_once = 50     #
#n_predict_once = 100
#n_predict_once = 200

# Run LSTM
x_axis_seq, predictions_over_time, run_data, KPI,  mid_data_over_time = LSTM(price_train_data_AS,price_test_data_AS,price_all_mid_data_AS,volume_train_data_AS,volume_test_data_AS,volume_all_mid_data_AS, D, num_unrollings, batch_size, num_nodes, n_layers, dropout, n_predict_once)

# =============================================================================
# Saving the results and finding the best epoch
# =============================================================================

# Now automatically chooses the one with the highest 'correct' 
best_prediction_epoch = PerformanceSaver(pp_data_price, run_data, KPI, n_predict_once, num_unrollings, batch_size)

# =============================================================================
# Visualisation of the results
# =============================================================================
#%%
#best_prediction_epoch = 4
plot = prediction(df, price_all_mid_data_AS, volume_all_mid_data_AS, x_axis_seq, predictions_over_time, best_prediction_epoch)


