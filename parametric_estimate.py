import numpy as np


def calculate_parametric(autocovariances):
    '''Get parametric IV estimators.
    
    Parameters
    autocovariances: 2D array of autocovariances
    where number of rows is number of lags and number of columns is number of samples'''

    # Maybe add days later

    # sample from price and time according to the sampling frequency
    RVsample = parametric_rv_calc(autocovariances)

    # calculate volatility estimators for each model
    RV_standard = np.array(RVsample.standard_RV_calc())
    restricted_learning = np.array(RVsample.restricted_learning_calc())
    learning_nonstrategic_noisyinfo = np.array(RVsample.learning_nonstrategic_noisyinfo_robust_calc())
    learning_nonstrategic_informed = np.array(RVsample.learning_nonstrategic_informed_calc())
    learning_strategic_informed = np.array(RVsample.learning_strategic_informed_calc())

    return [RV_standard, restricted_learning, learning_nonstrategic_noisyinfo, learning_nonstrategic_informed, learning_strategic_informed]

class parametric_rv_calc():
    '''Class to calculate intraday parametric IV estimators.
    
    Parameters
    autocovariances: 2D array of autocovariances
    max_lag: maximum calculated autocovariance lag'''

    def __init__(self, autocovariances, max_lag = 20):

        #autocovariances is a 2D array, with the first entry in each row being the variance
        self.autocovariances = autocovariances[:, 1:]
        self.var = autocovariances[:, 0]
        self.max_lag = max_lag

    def standard_RV_calc(self):
        '''Standard RV estimator.
        
        Returns sum of squared market returns over interval.'''
        rv = self.var 

        #multiply by 100^2 to get percentage variance
        return rv * (100 ** 2)
        
    def restricted_learning_calc(self):
        '''Restricted Learning IV estimator, absence of exogenous noise.
        
        Returns scaled RV.'''
        # learning rate equal to variance over first autocovariance, use magnitude to avoid complex numbers
        learning_rate = np.log(np.divide(self.var,abs(self.autocovariances[:, 0])))

        # var*(var+first_autocovariance)/(var-first_autocovariance) for each column
        rv = np.multiply(self.var,np.divide((self.var + self.autocovariances[:, 0]),(self.var - self.autocovariances[:, 0])))
        return rv * (100 ** 2)

    def learning_nonstrategic_noisyinfo_robust_calc(self):
        '''Estimator for only presence of non-strategic incompletely informed traders.

        Returns RV + 2*Adjusting Term.
        '''
        number_autocovariances = min(30, self.max_lag - 1)
        robust_term = np.empty((number_autocovariances, self.autocovariances.shape[0]))
        for i in range(number_autocovariances):
            robust_term[i, :] = np.divide(self.autocovariances[:,i],(self.autocovariances[:, i] - self.autocovariances[:, i + 1]))

        robust_term = np.median(robust_term, axis=0) # get median along each row, per column

        rv = self.var + 2 * np.multiply(self.autocovariances[:, 0], robust_term)
        return rv * (100 ** 2)

    def learning_nonstrategic_informed_calc(self):
        '''Estimator for only presence of non-strategic informed traders.
        
        Returns RV + 2*First-lag Autocovariance + 2*Adjusting Term.'''

        number_autocovariances = min(31, self.max_lag - 1)
        robust_term = np.empty((number_autocovariances, self.autocovariances.shape[0]))

        for i in range(1, number_autocovariances):
            robust_term[i - 1, :] = np.divide(self.autocovariances[:, i],(self.autocovariances[:, i] - self.autocovariances[:, i+1]))

        robust_term = np.median(robust_term, axis=0)

        rv = self.var + 2 * self.autocovariances[:, 0] + 2 * np.multiply(self.autocovariances[:, 1], robust_term)
        return rv * (100 ** 2)

    def learning_strategic_informed_calc(self):
        '''Estimator for presence of strategic informed traders.
        
        Returns RV + Private Information Scaling 1 * First-lag Autocovariance 
        + Private Information Scaling 2 * Second-lag Autocovariance.'''
        # S-period private information
        #print('Duration of strategic trading estimate is based on', self.max_lag, 'autocorrelations. ')

        # Calculate S according to estimator formula (see just above section 6.2)
        # sac and tmp are placeholder variables
        sac = self.autocovariances.sum(axis=1)
        sac = 2 * np.divide(sac,(self.autocovariances[:, 1]-self.autocovariances[:, 0]))
        tmp = np.divide((3 * self.autocovariances[:, 0] - self.autocovariances[:, 1]),(2 * (self.autocovariances[:, 1]-self.autocovariances[:, 0])))
            
        S   = np.sqrt(sac + np.square(tmp)) - tmp

        # replace nan indices with 0
        nan_indices = np.argwhere(np.isnan(S))
        S[nan_indices] = 0

        # Note: in the original code there is an error: S-3 instead of 3-S, as stated correctly in the supplementary derivation.
        print('Strategic trading is estimated to last', S, 'period(s).')
        rv = self.var + np.multiply(np.multiply(S,(3 - S)),self.autocovariances[:, 0]) + np.multiply(np.multiply(S,(S - 1)),self.autocovariances[:, 1])

        # replace nan indices values with infinite sum of autocovariances
        if nan_indices.size > 0:
        #    print(f'Unable to solve for duration of strategic trading (complex number). Model does not fit for {nan_indices}. Using infinite sum of ACs instead.')
            rv[nan_indices] = self.var[nan_indices] + 2 * np.sum(self.autocovariances,axis=1)[nan_indices]

        return rv * (100 ** 2)
