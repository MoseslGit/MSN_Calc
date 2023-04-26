import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt

import kernel_estimate
import parametric_estimate
import remedi_estimate
import preaveraging_estimate
import flat_top_estimate

def convert_csv_to_parquet(csv_file_name, parquet_file_name):
    '''Converts a csv file to a parquet file'''
    df = pd.read_csv(csv_file_name)

    # convert date in yyyy-mm--dd into yyyymmdd
    df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y%m%d')
    df.to_parquet(parquet_file_name, compression=None)

def load_day_data(data, date):
    '''Load data for a given day
    
    Parameters
    data: pandas dataframe of raw data
    date: date to load data for

    Returns raw price and timestamp data for a given day'''

    # Keep only the rows where date is the given date and price is non-zero
    data = data[data['DATE'] == date]
    data = data[data['PRICE'] != 0]

    data = data.reset_index(drop=True)

    # Convert TIME_M to datetime
    data['TIME_M'] = pd.to_datetime(data['TIME_M'], format='%H:%M:%S.%f')

    # Convert time to seconds
    data['ts'] = data['TIME_M'].dt.hour*3600 + data['TIME_M'].dt.minute*60 + data['TIME_M'].dt.second

    # keep only the rows between 9:30 and 16:00
    data = data[(data['ts'] >= 9*3600 + 30*60) & (data['ts'] <= 16*3600)]

    # Prices will be converted to log price in each individual function
    time = data['ts']
    price = data['PRICE']

    # create secondary price and time series sampled every second
    second_sampled_price = data['PRICE'].reindex(range(9*3600 + 30*60, 16*3600 + 1), method = 'ffill')
    second_sampled_time = np.arange(9*3600 + 30*60, 16*3600 + 1)

    return price, time, second_sampled_price, second_sampled_time

def get_sampled_returns(sampled_price):
    '''Calculate log returns for a given price series.'''

    # get log returns
    price_percent_change = sampled_price.pct_change()
    pct_returns = np.log(1 + price_percent_change)

    sampled_returns = pct_returns[~np.isnan(pct_returns)]
    sampled_returns = sampled_returns.reset_index(drop=True)

    return sampled_returns

def autocovariance_calc(sampled_returns, max_lag, sampling_frequency):
    '''Calculate autocovariance for a given price series and sampling frequencies.
    
    Parameters
    sampled_price: price series to calculate autocovariance
    max_lag: maximum autocovariance lag to calculate
    sampling_frequency: sampling frequency
    
    Returns autocovariance series'''

    print(f'sampled_returns_length for sampling freq {sampling_frequency} = {len(sampled_returns)}')
    print(f'average pct diff = {np.mean(sampled_returns)}')

    # variance is a special case of autocovariance with lag = 0
    autocovariance = smt.acovf(sampled_returns, nlag = max_lag, fft = True)*len(sampled_returns)
    return autocovariance


def main(data_file_name):
    '''Main function to calculate Volatility Estimators for different models and sampling frequencies.
    
    Parameters
    data_file_name: name of file to load data from
    
    Calls
    a_statistical_estimate.calculate_statistical
    a_parametric_estimate.calculate_parametric
    a_remedi_estimate.calculate_remedi
    a_preaveraging_estimate.calculate_preaveraging
    a_flat_top_estimate.calculate_flat_top
    a_plot_results.plot_results
    
    Outputs plots of estimated IVs for each model and sampling frequency'''

    data = pd.read_parquet(data_file_name, engine="fastparquet") 

    # Create array of unique days
    data['DATE'] = pd.to_datetime(data['DATE'].astype(str), format='%Y%m%d').dt.date
    unique_days = []
    for date in data['DATE'].unique():
        unique_days.append(date)

    # Set sampling frequencies: 1, 2, 3, ..., 10, 20, 30, ..., 600, 660, 720, ..., 1800
    #sampling_frequencies = np.concatenate((np.arange(1,11), np.arange(20,121,10), np.arange(150,601,30), np.arange(660,841,60), np.arange(960,1801,120)), axis = None)

    # Alternative to above: 1, 2, 3, ..., 10, 20, 30, ..., 120
    # Used when there are not enough samples in a day for lower sampling frequencies
    sampling_frequencies = np.concatenate((np.arange(1,11), np.arange(20,121,10)), axis = None)

    # Add 0 to the end of sampling_frequencies to hold second-sampled data
    sampling_frequencies = np.append(sampling_frequencies, 0)

    # Define dataframe to store first entry in IVs each day
    IV_intraday_df = pd.DataFrame(columns = ['date', 'standard', 'rectangular_kernel', 
                                             'rectangular_triangular_kernel', 'restricted_learning', 
                                             'learning_nonstrategic_noisyinfo', 'learning_nonstrategic_informed', 
                                             'learning_strategic_informed','PVG', 'flat_top'])

    for i, date in enumerate(unique_days):
        print(date)
        price, time, second_sampled_price, second_sampled_time = load_day_data(data, date)

        # We use max_lag = 100 to provide enough autocovariance values for the flat-top estimator
        # Diebold and Strasser (2013) use "j as max(4, min(j, 100))"" where original j is number of entries. 
        max_lag = 100

        # Create empty arrays to store autocovariances and IVs

        # include variance in first position, so add 1 to max_lag for length
        # We only take 20 autocovariances for the other estimators, but this can be extended
        autocovariances = np.zeros((len(sampling_frequencies), max_lag + 1), dtype = float) 
        remedi_acov = np.zeros((len(sampling_frequencies), 21), dtype = float)
        #local_averaging_acov = np.zeros((len(sampling_frequencies), 21), dtype = float)

        RV_PVG = np.zeros(sampling_frequencies.shape[0], dtype = float)
        RV_flat_top = np.zeros(sampling_frequencies.shape[0], dtype = float)

        for index, sampling_frequency in enumerate(sampling_frequencies):

            # Special case for second-sampled data
            if sampling_frequency == 0:
                raw_sampled_price = second_sampled_price
                sampled_time = second_sampled_time
            else:
                raw_sampled_price = price[::sampling_frequency]
                sampled_time = np.array(time[::sampling_frequency])

            # note that when we use log sampled returns, we multiple the result by 100**2 to account for percentages
            log_sampled_returns = get_sampled_returns(raw_sampled_price)
            log_sampled_price = np.array(np.log(raw_sampled_price))

            # Create a matrix of autocovariances to speed up calculations for estimators in Diebold and Strasser (2013)
            autocovariances[index] = autocovariance_calc(log_sampled_returns, max_lag, sampling_frequency)

            # Calculate alternative autocovariance estimators according to Li and Linton (2022) 
            remedi_acov[index]= remedi_estimate.calculate_remedi(log_sampled_price, sampled_time, date)
            
            # Calculate individual intraday IV estimators from Varneskov (2016) and Jacod et al. (2019)
            RV_flat_top[index] = flat_top_estimate.calculate_fFTRK(log_sampled_returns, 25, 0.5)
            RV_PVG[index] = preaveraging_estimate.calculate_pvg(log_sampled_price) * (100 ** 2)

        # Calculate statistical and parametric estimators as in Diebold & Strasser (2013)
        DS_parametric_rv = parametric_estimate.calculate_parametric(autocovariances)
        DS_statistical_rv = kernel_estimate.calculate_statistical(autocovariances)

        # Store IV estimators for each model
        IV_sampling_freq_df = pd.DataFrame(data = [sampling_frequencies, DS_parametric_rv[0], DS_statistical_rv[0], DS_statistical_rv[1], DS_parametric_rv[1], DS_parametric_rv[2], DS_parametric_rv[3], DS_parametric_rv[4], RV_PVG, RV_flat_top]).T
        IV_sampling_freq_df.columns = ['sampling_frequencies', 'standard', 'rectangular_kernel', 'rectangular_triangular_kernel', 'restricted_learning', 'learning_nonstrategic_noisyinfo', 'learning_nonstrategic_informed', 'learning_strategic_informed','PVG','flat_top']
        IV_sampling_freq_df.to_csv(f'IV_sampling_freq_df_{date}{data_file_name}.csv')

        # Store autocovariance estimates
        autocovariance_df = pd.DataFrame(data = autocovariances, columns = [f'lag_{i}' for i in range(autocovariances.shape[1])])
        autocovariance_df['sampling_frequencies'] = sampling_frequencies
        remedi_autocovariance_df = pd.DataFrame(data = remedi_acov, columns = [f'lag_{i}' for i in range(remedi_acov.shape[1])])
        remedi_autocovariance_df['sampling_frequencies'] = sampling_frequencies
    
        autocovariance_df.to_csv(f'autocovariance_df_{date}{data_file_name}.csv')
        remedi_autocovariance_df.to_csv(f'remedi_autocovariance_df_{date}{data_file_name}.csv')

        # Store first entry in RVs for each day
        IV_intraday_df.loc[i] = [date, DS_parametric_rv[0][0], DS_statistical_rv[0][0], DS_statistical_rv[1][0], DS_parametric_rv[1][0], DS_parametric_rv[2][0], DS_parametric_rv[3][0], DS_parametric_rv[4][0], RV_PVG[0], RV_flat_top[0]]
    
    # Save dataframe to csv
    IV_intraday_df.to_csv(f'IV_intraday_df {data_file_name}.csv')



