import numpy as np
import matplotlib.pyplot as plt


# Note that this Python version assumes that the input Y is a 1D numpy array and ts is also a 1D numpy array. 
# The input pars is a dictionary with keys phi_n, kn, in, and l.
# Not only keeping between (9:30 and 16:00 in matlab)

def AVarReMeDI2(Y, ts, pars):
    '''Calculate the AVarReMeDI2 statistic for a given time series Y and transaction times ts.'''
    # Extract the tuning parameters
    p_n = pars['phi_n']
    k_n = pars['kn']
    i_n = pars['in']
    l = pars['l']
    n = len(ts)
    
    # Normalize the transaction times
    ts = ts-ts[0]
    ts = ts/ts[-1]
    
    # Calculate the obvservation time based statistics 
    diff1 = ts[1:] - ts[:-1]

    diffknT = ts[k_n:] - ts[:-k_n]

    dn1 = k_n * diff1[1+k_n:n-k_n-1] - diffknT[k_n+2:n-k_n]
    dn2 = np.divide(dn1, np.maximum(p_n, diffknT[:n-2-2*k_n]))
    dn = np.square(dn2)

    # take the kn, 2kn and 3kn differences of log-prices
    diffknY = Y[:n-k_n] - Y[k_n:n]
    diff2knY = Y[2*k_n:n] - Y[:n-2*k_n]
    diff3knY = Y[3*k_n:n] - Y[:n-3*k_n]
    Rj = sum(np.multiply(diffknY[2*k_n+l:n-k_n], diff2knY[:n-l-3*k_n]))


    # U(1)
    U1 = np.sum(dn)
    
    # U(2)
    U2 = sum(np.multiply(diff2knY[2+3*k_n:n-l-3*k_n], np.multiply(diffknY[2+5*k_n+l:n-k_n], dn[0:n-2-6*k_n-l])))

    # U(3)
    U3 = np.sum(np.multiply(diffknY[2+5*k_n+l:n-l-5*k_n], diffknY[2+9*k_n+2*l:n-k_n]) * np.multiply(diff2knY[2+3*k_n:n-7*k_n-2*l], diff2knY[2+7*k_n+l:n-3*k_n-l]) * dn[:n-10*k_n-2*l-2])

    U4 = -sum(np.multiply(diffknY[1+2*k_n+l:n-l-5*k_n],
                        np.multiply(diffknY[1+6*k_n+2*l:n-k_n],
                                    np.multiply(diff2knY[1:n-7*k_n-2*l],
                                                diff2knY[1+4*k_n+l:n-3*k_n-l]))))

    S1 = sum(np.multiply(diffknY[6*k_n:n-5*k_n-2*l],
                        np.multiply(diffknY[10*k_n+2*l:n-k_n],
                                    np.multiply(diff2knY[8*k_n+2*l:n-3*k_n],
                                                diff2knY[8*k_n+l:n-3*k_n-l])))) + \
        sum(np.multiply(diffknY[1+3*k_n+l:n-k_n],
                        np.multiply(diffknY[1+3*k_n:n-k_n-l],
                                    np.multiply(diff2knY[1+k_n+l:n-3*k_n],
                                                diff3knY[1:n-4*k_n-l])))) + U4
    for k in range(1, i_n+1):
        Uk1 = sum(np.multiply(diffknY[6*k_n+2*k:n-5*k_n-k-2*l],
                            np.multiply(diffknY[10*k_n+2*l+3*k:n-k_n],
                                        np.multiply(diff2knY[8*k_n+2*l+3*k:n-3*k_n],
                                                    diff2knY[8*k_n+l+2*k:n-3*k_n-k-l]))))

        Uk2 = sum(np.multiply(diffknY[1+3*k_n+l+k:n-k_n],
                            np.multiply(diffknY[1+3*k_n+k:n-k_n-l],
                                        np.multiply(diff2knY[1+k_n+l:n-3*k_n-k],
                                                    diff3knY[1:n-4*k_n-k-l]))))

        S1 += 2*(3*Uk1+Uk2+U4)

    S2 = U3
    Rj = Rj/n
    S3 = (Rj**2*U1)-2*Rj*U2

    # Calculate the asymptotic variance
    avar = (S1+S2+S3)/n
    avar = max(-avar, avar)
    
    # Create the stats output
    stats = {
        'Rj': Rj,
        'avar': avar
    }
    
    return stats


def calculate_remedi(price, time, date):
    '''Calculate autocovariance using ReMeDI and Local Averaging Methods
    
    Parameters
    price: log price
    time: time
    date: date
    
    Returns
    ReMeDIest: ReMeDI estimator
    covLA: Local Averaging estimator
    '''
    # Set the parameters
    covorder = 20
    n = len(price)
    pars = {'kn': 10, 'phi_n': (10**(3/5))/n, 'in': 5}

    # Initialize the ReMeDI estimators and their asymptotic variance
    ReMeDIest = np.zeros(covorder + 1)
    ReMeDIavar = np.zeros(covorder + 1)

    # Set the LA parameter
    knLA = 6

    # Loop over the lags
    for l in range(covorder + 1):
        pars['l'] = l
        stats = AVarReMeDI2(price, time, pars)
        ReMeDIest[l] = stats['Rj']
        ReMeDIavar[l] = stats['avar']

    return ReMeDIest