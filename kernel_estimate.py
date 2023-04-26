import numpy as np

def calculate_statistical(autocovariances):
    '''Get statistical IV estimators in Diebold & Strasser (2013).
    
    Parameters
    autocovariances: 2D array of autocovariances
    where number of rows is number of lags and number of columns is number of samples'''

    RVsample = stat_rv_calc(autocovariances)
    RV_rectangular_kernel = np.array(RVsample.RV_rectangular_kernel_calc())
    RV_rectangular_triangular_kernel = np.array(RVsample.RV_rectangular_triangular_kernel_calc())

    return [RV_rectangular_kernel, RV_rectangular_triangular_kernel]

class stat_rv_calc():
    '''Class to calculate intraday statistical IV estimators.
    
    Parameters
    autocovariances: 2D array of autocovariances
    lag_window: lag window for kernel
    max_lag: maximum calculated autocovariance lag'''
    def __init__(self, autocovariances, lag_window = 0, max_lag = 20):

        #autocovariances is a 2D array, with the first entry in each row being the variance
        self.autocovariances = autocovariances[:,1:]
        self.var = autocovariances[:,0]

    def RV_standard_calc(self):
        '''Standard RV estimator.'''
        rv = self.var 
        
        # realized variance here (rv) is defined as realized volatility in the original paper: sum of squared market returns over interval #multiply by 100^2 for percentage
        return rv*(100**2) #multiply by 100^2 for percentage
        
    def RV_rectangular_kernel_calc(self): 
        '''Rectangular Kernel estimator, where lag_window = 1 case is bid-ask estimator of Roll 1984'''
        self.lag_window = 1
        rv = self.var + 2*np.sum(self.autocovariances[:,0:self.lag_window], axis=1)
        return rv*(100**2)

    def RV_rectangular_triangular_kernel_calc(self):
        '''Rectangular-Triangular Kernel estimator. See Hansen and Lunde 2006.'''

        self.lag_window = 30

        if self.autocovariances.shape[1] < 2*self.lag_window:
            print('RVacnwHL: Insufficient number of autocovariances.')
            rv = np.nan
            return rv
        # Rectangular Kernel Component
        rv = self.var + 2*np.sum(self.autocovariances[:,0:self.lag_window], axis=1)
        # Triangular Kernel Component
        for i in range(self.lag_window):
            rv += 2*((self.lag_window-i)/self.lag_window)*self.autocovariances[:,self.lag_window+i]

        return rv*(100**2)



