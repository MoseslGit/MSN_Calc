import pandas as pd
import matplotlib.pyplot as plt
# This file reads saved csv files and plots them

def read_csv_to_df(csv_file):
    '''Read csv file to dataframe'''

    df = pd.read_csv(csv_file)
    return df

def plot_week_IV(df, stock, date):
    '''Plot all intraday volatility estimates for a given stock earnings week

    Parameters
    df: dataframe of intraday volatility estimates
    stock: stock ticker
    date: earnings date
    '''

    fig = plt.figure(figsize=(10, 6))

    marker_style = {'standard': 'o', 'bid_ask': '^', 
                'restricted_learning': 'v', 'learning_nonstrategic_noisyinfo': 'P',
                'learning_nonstrategic_informed': 'X', 'learning_strategic_informed': '*', 
                'preaveraging': 'D', 'flat_top': '+', 'standard_statistical': 's'}

    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'grey']
    
    # Plot each series of data
    for col in df.columns:
        if col != 'date':
            plt.plot(df['date'], df[col], label=col, marker=marker_style[col], markersize=8, color=colors.pop(0))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add axis labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volatility', fontsize=14)
    plt.title(f'Intraday Volatility estimates earnings week for {stock} on {date}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f'IV_Est/Intraday_estimates_{date}_{stock}.png', dpi=300, bbox_inches='tight')

    plt.close('all')

def plot_week_IV_selected(df, stock, date):
    '''Plot selected intraday volatility estimates for a given stock earnings week
    
    Parameters
    df: dataframe of intraday volatility estimates
    stock: stock ticker
    date: earnings date
    '''

    fig = plt.figure(figsize=(10, 6))

    # drop standard, restricted learning, and learning nonstrategic noisyinfo
    df = df.drop(['standard', 'restricted_learning', 'learning_nonstrategic_noisyinfo'], axis=1)

    marker_style = {'bid_ask': '^', 'learning_nonstrategic_informed': 'X',
                    'learning_strategic_informed': '*', 'preaveraging': 'D',
                    'flat_top': '+', 'standard_statistical': 's'}
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'grey']
    
    # Plot each series of data
    for col in df.columns:
        if col != 'date':
            plt.plot(df['date'], df[col], label=col, marker=marker_style[col], markersize=8, color=colors.pop(0))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add axis labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volatility', fontsize=14)
    plt.title(f'Selected Intraday Volatility estimates earnings week for {stock} on {date}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f'IV_Est/Select_intraday_estimates_{date}_{stock}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_noise_captured(df, stock, date, noise_type):
    '''Plot noise captured for a given stock and statistical baseline
    
    Parameters
    df: dataframe of intraday volatility estimates
    stock: stock ticker
    date: earnings date
    noise_type: either standard_statistical, flat_top, or preaveraging
    '''
    # split the data frame, with the other having the last 3 columns
    df1 = df.iloc[:, :-3]
    df2 = df.iloc[:, -3:]

    for i in range (2,7):
        df1.iloc[:, i] = (df1.iloc[:, 1] - df1.iloc[:, i])/(df1.iloc[:, 1] - df2.loc[:, noise_type])
    fig = plt.figure(figsize=(10, 6))

    marker_style = {'bid_ask': '^', 
                'restricted_learning': 'v', 'learning_nonstrategic_noisyinfo': 'P',
                'learning_nonstrategic_informed': 'X', 'learning_strategic_informed': '*'}

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for col in df1.columns[2:7]:
        plt.plot(df1['date'], df1[col], label=col, marker=marker_style[col], markersize=8, color=colors.pop(0))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add axis labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('% Noise Captured', fontsize=14)
    plt.title(f'% Noise Captured relative to {noise_type} for {stock} on {date}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f'Noise_Cap/Noise_captured_{date}_{stock}_{noise_type}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    df1.to_csv(f'Noise_Cap/Noise_captured_{date}_{stock}_{noise_type}.csv', index=False)

def plot_noise_comparison(df, stock, date):
    '''Plot a comparison of the noise captured with different statistical 
    baselines against learning_nonstrategic_informed
    
    Parameters
    df: dataframe of intraday volatility estimates
    stock: stock ticker
    date: earnings date
    '''
    # split the data frame, with the other having the last 3 columns
    df1 = df.iloc[:, :-3]
    df2 = df.iloc[:, -3:]

    for col in df2.columns:
        df2.loc[:, col] = (df1.iloc[:, 1] - df1.loc[:, 'learning_nonstrategic_informed'])/(df1.iloc[:, 1] - df2.loc[:, col])
    fig = plt.figure(figsize=(10, 6))

    # plot df2 against date
    for col in df2.columns:
        plt.plot(df1['date'], df2[col], label=col, marker='o', markersize=8)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add axis labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('% Noise Captured', fontsize=14)
    plt.title(f'% Noise Captured by Learning NonStrategic Informed estimator for {stock} on {date}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f'Noise_Cap/Noise_comp_{date}_{stock}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    df2.to_csv(f'Noise_Cap/Noise_comp_{date}_{stock}.csv', index=False)

def plot_autocovariance_comparison(standard_df, remedi_df, date, stock):
    '''Plot both autocovariance estimates on the same graph
    
    Parameters
    standard_df: dataframe of standard autocovariance estimates
    remedi_df: dataframe of ReMeDI autocovariance estimates
    date: earnings date
    stock: stock ticker
    '''

    fig = plt.figure(figsize=(10, 6))

    # keep only first 21 columns
    standard_df = standard_df.iloc[:, 3:22]
    remedi_df = remedi_df.iloc[:, 3:22]

    # get as series
    standard = standard_df.iloc[0].tolist()
    remedi = remedi_df.iloc[0].tolist()

    # Plot both series of data
    plt.plot(standard, label='Standard', marker='o', markersize=8)
    plt.plot(remedi, label='ReMeDI', marker='^', markersize=8)

    plt.legend()
    plt.xlabel('Lags', fontsize=14)
    plt.ylabel('Autocovariance', fontsize=14)
    plt.title(f'Standard and ReMeDI autocovariance estimates for {stock} on {date}', fontsize=16)
    plt.savefig(f'AC/Autocovariance_comparison_{date}_{stock}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_autocovariance_sampling_freqs(type, df, date, stock):
    '''Plot autocovariance estimates from different sampling frequencies

    Parameters
    type: either standard or remedi
    df: dataframe of autocovariance estimates
    date: earnings date
    stock: stock ticker
    '''
    fig = plt.figure(figsize=(10, 6))
    # Get every 4th row of the dataframe
    df = df.iloc[::4]
    
    # store sampling frequency column in a list
    sampling_freqs = df['sampling_frequencies'].tolist()
    df = df.drop('sampling_frequencies', axis=1)

    # keep only first 21 columns
    df = df.iloc[:, 1:22]
    df = df.T
    df = df.reset_index(drop=True)

    # Plot each series of data
    for index, col in enumerate(df.columns):
        plt.plot(df[col], label=sampling_freqs[index], marker='o', markersize=8)

    plt.legend()

    # Add axis labels and title
    plt.xlabel('Lags', fontsize=14)
    plt.ylabel('Autocovariance', fontsize=14)
    plt.title(f'{type} autocovariance estimates for {stock} on {date}', fontsize=16)
    plt.savefig(f'AC/Sample_freq_autocovariance_{type}_{date}_{stock}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_volatility_signature(df, date, stock):
    '''Plot Volatility Signature for a given day.
    
    Parameters
    df: dataframe of volatility estimates for different sampling frequencies
    date: earnings date
    stock: stock ticker
    '''

    fig = plt.figure(figsize=(10, 6))
    df = df.iloc[:-1, 1:]

    df = df.rename(columns={'PVG': 'preaveraging'})
    df = df.rename(columns={'rectangular_kernel': 'bid_ask'})

    # Move rectangular_triangular_kernel to the last column
    cols = list(df.columns.values)
    cols.pop(cols.index('rectangular_triangular_kernel'))
    df = df[cols+['rectangular_triangular_kernel']]
    df = df.rename(columns={'rectangular_triangular_kernel': 'standard_statistical'})

    marker_style = {'standard': 'o', 'bid_ask': '^', 
                'restricted_learning': 'v', 'learning_nonstrategic_noisyinfo': 'P',
                'learning_nonstrategic_informed': 'X', 'learning_strategic_informed': '*', 
                'preaveraging': 'D', 'flat_top': '+', 'standard_statistical': 's'}

    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'grey']

    plt.xscale('log')
    # Plot each series of data
    for col in df.columns:
        if col != 'sampling_frequencies':
            plt.plot(df['sampling_frequencies'], df[col], label=col, marker = marker_style[col], markersize=8, color=colors.pop(0))

    plt.legend()

    # Add axis labels and title
    plt.xlabel('Inverse of Sampling Frequency in Ticks', fontsize=14)
    plt.ylabel('Volatility Estimate', fontsize=14)
    plt.title(f'Volatility Signature for {stock} on {date}', fontsize=16)
    plt.savefig(f'Vol_Sig/Vol_sig_{date}_{stock}.png', dpi=300, bbox_inches='tight')
    plt.close('all')


# Example of how to use the functions
stock = 'AAPL'
datafile_name = 'AAPL_14102021_trade.parquet' 
earnings_date = '2021-10-14'

intraday_data = read_csv_to_df(f'IV_intraday_df {datafile_name}.csv')

intraday_data = intraday_data.rename(columns={'PVG': 'preaveraging'})
intraday_data = intraday_data.rename(columns={'rectangular_kernel': 'bid_ask'})

# Move rectangular_triangular_kernel to the last column
cols = list(intraday_data.columns.values)
cols.pop(cols.index('rectangular_triangular_kernel'))
intraday_data = intraday_data[cols+['rectangular_triangular_kernel']]
intraday_data = intraday_data.rename(columns={'rectangular_triangular_kernel': 'standard_statistical'})

intraday_data.drop(intraday_data.filter(regex="Unname"),axis=1, inplace=True)

unique_days = []
for date in intraday_data['date'].unique():
    unique_days.append(date)

print(unique_days)

plot_week_IV(intraday_data, stock, earnings_date)
plot_week_IV_selected(intraday_data, stock, earnings_date)
plot_noise_captured(intraday_data, stock, earnings_date, 'preaveraging')
plot_noise_captured(intraday_data, stock, earnings_date, 'flat_top')
plot_noise_captured(intraday_data, stock, earnings_date, 'standard_statistical')
plot_noise_comparison(intraday_data, stock, earnings_date)

for date in unique_days:
    plot_autocovariance_comparison(read_csv_to_df(f'autocovariance_df_{date}{datafile_name}.csv'), 
                                read_csv_to_df(f'remedi_autocovariance_df_{date}{datafile_name}.csv'), 
                                date, stock)
    plot_autocovariance_sampling_freqs('Standard', read_csv_to_df(f'autocovariance_df_{date}{datafile_name}.csv'),
                                       date, stock)
    plot_autocovariance_sampling_freqs('ReMeDI', read_csv_to_df(f'remedi_autocovariance_df_{date}{datafile_name}.csv'),
                                        date, stock)
    plot_volatility_signature(read_csv_to_df(f'IV_sampling_freq_df_{date}{datafile_name}.csv'), date, stock)