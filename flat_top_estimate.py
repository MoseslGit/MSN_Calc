import numpy as np

# In the implementation code, estimator bandwidth takes either 25, 50, or 75
# fc is (rkn(l)*fH^0.6)/T, where rkn takes the values 0.5, 1, or 1.5
# number of assets = 1

def calculate_fFTRK(returns, estimator_bandwidth, fc):
    '''Calculate the Flat Top Realized Kernel Estimator as Varneskov (2016)
    
    Parameters
    returns: a numpy array of log returns
    estimator_bandwidth: an integer, either 25, 50, or 75
    fc: a float, (rkn(l)*fH^0.6)/T, where rkn takes the values 0.5, 1, or 1.5
    
    Returns
    Flat Top Realized Kernel Estimator'''

    number_of_assets = 1
    fgamma0 = 0
    fgammaplus = 0
    fgammaminus = 0

    for fh in range(estimator_bandwidth + int(np.floor(fc * estimator_bandwidth))):
        fx = fh / estimator_bandwidth
        # Flat-top Parzen Window
        if fx <= fc:
            fweight = 1

        if fx > fc:
            fx1 = fx - fc
            if fx1 - 0.5 <= 0:
                fweight = 1 - 6 * fx1 ** 2 + 6 * fx1 ** 3
            if fx1 - 0.5 > 0:
                fweight = 2 * (1 - fx1) ** 3

        fgammaplush = 0
        fgammaminush = 0

        for fi in range(fh, len(returns)):
            if fh == 0:
                # we only consider one asset at a time, so use square
                # outer product?
                fgamma0 += np.multiply(returns[fi], returns[fi])
            if fh > 0:
                fgammaplus += fweight * np.multiply(returns[fi], returns[fi - fh])
                fgammaminus += fweight * np.multiply(returns[fi - fh], returns[fi])
    
    fgammaplus += fweight * fgammaplush
    fgammaminus += fweight * fgammaminush

    fFTRK = fgamma0 + fgammaplus + fgammaminus

    return fFTRK*(100**2)
