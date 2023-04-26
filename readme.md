# Source Code for "Earnings Events and Market Microstructure Noise: Fitting Parametric Models to Statistical Bases"

## Instructions

Call main function on a parquet file, or call the convert_csv_to_parquet prior to the main function.

Examples:

**main('trade_data.parquet')**

OR

**convert_csv_to_parquet('trade_data.csv')**

**main('trade_data.parquet')**

This will output 4 CSV files, which can be used directly or with plot_csv.py to visualize data.

'IV_sampling_freq_df_{date}{data_file_name}.csv' - Contains IV estimates for a certain earnings week at a range of tick-time sampling frequencies specified in main.py. The last sampling frequency (21 here) is sampled every second instead.

'autocovariance_df_{date}{data_file_name}.csv' - Standard autocovariance estimates for different sampling frequencies, up to 20 lags (this can be adjusted).

'remedi_autocovariance_df_{date}{data_file_name}.csv' - ReMeDI autocovariance estimates for different sampling frequencies.

'IV_intraday_df {data_file_name}.csv' - Stores only IV estimates from the 1-tick-time calculations for each day across the week

## Input Format

The main function accepts a parquet file with at least three columns, which can be downloaded from the WRDS TAQ database: 'DATE', 'PRICE' (transaction price), 'TIME_M' (trade timestamp).

## Implementation Details

Each volatility estimate is categorized into different files:

parametric_estimate.py and kernel_estimate.py include all the estimators considered in Diebold and Strasser (2013).

remedi_estimate.py corresponds to Li & Linton (2022).

flat_top_estimate.py corresponds to Varneskov (2016).

preaveraging_estimate.py corresponds to Jacod et al. (2019).

Unless otherwise mentioned, tuning parameters follow the original paper, with no optimization.
