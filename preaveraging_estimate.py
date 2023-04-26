import numpy as np
import numpy.matlib
import pandas as pd


# PVG_var here is IV/T, which multiplies the result by 252, the number of trading days in a year
# Here we calculate RV over a day - multiply final results together in main

# weight function g
def g(x):
    '''Simple kernel weighting function.'''
    return np.minimum(x, (1 - x))

# Original function in MATLAB file by Da and Xiu (2021). Unused parts are multi-line commented out
#def PreAvg_JLZ(Y, J, T, knl, hnl, knpl):
# Key: Y is returns
# J is the number of autocorrelations
# T is 1/252 (trading days in a year)
#knl, hnl, knpl are tuning parameters - set when calling the function
# in the original code implementation, 27 different parameter combinations are tried
# and the one with the best RMSE is chosen to present
# here we follow the parameters in the original 2019 paper

def calculate_pvg(Y):
    '''Calculate the pre-averaging volatility estimator as in Jacod et al. (2017).
    
    Parameters
    Y: numpy array of log returns'''
    N = len(Y)

    # From Jacod et al. 2019, we set tuning parameters as follows
    k_n = int(N ** (1/5))
    print(k_n)
    k_n_dash = int(N ** (1/8))
    print(k_n_dash)

    theta = 0.8
    rho = 0.7
    h_n = round(0.5 * np.sqrt(N)) #QMLE paper mentions this is 0.8 instead
    h_n_dash = round(theta * N ** rho) # renamed hn_p to h_n_dash for clarity

    # See section 3.1 in Jacod et al. 2019
    g_bar = np.zeros(h_n)

    # Calculate g_bar
    # Note that when accessing indices, use k
    # When accessing values, use k+1
    # This is due to differences in python and matlab indexing

    '''for k in range(h_n):
        g_bar[k] = g((k+1) / h_n) - g(k / h_n)'''

    # Changing the use of g weight function to use vectors instead of loops
    vfunc1 = np.vectorize(lambda x: g(x / h_n))
    vfunc2 = np.vectorize(lambda x: g((x - 1) / h_n))
    g_bar = vfunc1(np.arange(1, h_n + 1)) - vfunc2(np.arange(1, h_n + 1))

    # Alternative implementation
    #k_values = np.arange(1, h_n + 1)
    #g_values = np.minimum(k_values / h_n, (1 - k_values / h_n))
    #g_bar = np.diff(g_values)
    #g_bar = np.append(g_values[0], g_bar)

    # Sum g(i/h_n)**2 for i = 0 to h_n, with special case j = 0
    phi_n_0 = np.sum(g((np.arange(h_n + 1) / h_n) ** 2)) / h_n

    # Define array phi_bar (phi_bar_n_j according to the docs)
    phi_bar = np.zeros(k_n_dash + 1)
    for k in range(1, k_n_dash + 2):
        phi_bar[k - 1] = h_n * np.sum(g_bar[k - 1:h_n] * g_bar[:h_n - k + 1])


    '''g_bar_p = np.zeros(h_n_dash)

    for k in range(h_n_dash):
        g_bar_p[k] = g(k+1 / h_n_dash) - g((k) / h_n_dash)'''

    # using pandas rolling mean as the fastest alternative
    Y_bar = pd.Series(Y[:len(Y)]).rolling(window=k_n, center=False).mean().values

    # Remove the first k_n-1 elements of Y_bar
    Y_bar = Y_bar[k_n - 1:]

    # U(m) here is a proxy for r(m), autocovariances (See 2.8)
    U = np.zeros(k_n_dash + 1)
    # Taking ind out of the loop for more efficient computation
    ind = np.arange(0, N + 1 - 5 * k_n)

    # Might be able to use slicing here? In any case it loops 10 times so not important
    for k in range(k_n_dash + 1):
        U[k] = np.sum((Y[ind] - Y_bar[2 * k_n + ind]) * (Y[k + ind] - Y_bar[4 * k_n + ind])) #Eqn 3.5

    J2 = N + 1 - h_n
    Y_bar_hn = np.zeros(J2)

    vfunc = np.vectorize(lambda x: -np.sum(g_bar * Y[x:x+h_n]))
    Y_bar_hn = vfunc(np.arange(1, J2))

    # Alternative implementation
    # weighting function times returns for each lag
    #for k in range(1, J2):
    #    Y_bar_hn[k-1] = -np.sum(g_bar * Y[k:k+h_n])

    # Equation 3.8
    # h_n = h_bar_n

    IV = np.sum(Y_bar_hn ** 2) / (h_n * phi_n_0) - 1 / (h_n ** 2 * phi_n_0) * \
                             (phi_bar[0] * U[0] + 2 * np.sum(phi_bar[1:] * U[1:]))

    return IV
    # Quarticity calculation
    # Untested, use with caution
    '''
    ln = np.round(theta * N ** (2/3)).astype(int)
    J3 = N + 1 - ln

    Y_bar_ln = np.zeros(J3)
    g_bar_ln = np.zeros(ln)

    for k in range(1, ln + 1):
    g_bar_ln[k - 1] = g(k / ln) - g((k - 1) / ln)

    for k in range(1, J3 + 1):
    Y_bar_ln[k - 1] = -np.sum(g_bar_ln * Y[k:k+ln-1])

    Quarticity = np.sum(Y_bar_ln[:J3-ln]2 * Y_bar_ln[ln:J3]2) / phi_n_02 / T / ln2 / N
    
    R_est = np.zeros(J + 1)
    U_0 = np.zeros(J + 1)

    for j in range(J + 1):
    mu = j
    ind = np.arange(1, N + 1 - mu - 2 * 2 * kn)
    U_0[j] = np.sum((Y[ind] - Y_bar[mu + kn + ind]) * (Y[j + ind] - Y_bar[mu + 3 * kn + ind]))
    R_est[j] = U_0[j] / len(ind)

    r_est = R_est / R_est[0]

    UU_1 = np.zeros((kn_p+1, J+1))
    UU_2 = np.zeros((kn_p+1, J+1))
    UU_3 = np.zeros((kn_p+1, J+1))
    UU_4 = np.zeros((kn_p+1, J+1))

    for m in range(k_n_dash+1):
        for j in range(J+1):
            mu = j + m
            ind = np.arange(0, N+1-mu-2*4*kn)
            UU_1[m, j] = np.sum((Y[ind]-Y_bar[mu+kn+ind]) * (Y[j+ind]-Y_bar[mu+3*kn+ind]) * (Y[m+ind]-Y_bar[mu+5*kn+ind]) * (Y[j+m+ind]-Y_bar[mu+7*kn+ind]))
            UU_2[m, j] = np.sum((Y[m+ind]-Y_bar[mu+kn+ind]) * (Y[j+m+ind]-Y_bar[mu+3*kn+ind]) * (Y[ind]-Y_bar[mu+5*kn+ind]) * (Y[j+ind]-Y_bar[mu+7*kn+ind]))
            UU_3[m, j] = np.sum((Y[ind]-Y_bar[mu+kn+ind]) * (Y[j+ind]-Y_bar[mu+3*kn+ind]) * (Y[m+ind]-Y_bar[mu+5*kn+ind]) * (Y[m+ind]-Y_bar[mu+7*kn+ind]))
            UU_4[m, j] = np.sum((Y[m+ind]-Y_bar[mu+kn+ind]) * (Y[j+m+ind]-Y_bar[mu+3*kn+ind]) * (Y[ind]-Y_bar[mu+5*kn+ind]) * (Y[ind]-Y_bar[mu+7*kn+ind]))



    U_bar_1 = np.zeros(J + 1)
    U_bar_2 = np.zeros(J + 1)

    for j in range(J + 1):
        mu = j
        mu_pp = j + j
        
        ind = np.arange(N + 1 - mu_pp - (2*4 + 1)*kn)
        
        U_bar_1[j] = np.sum((Y[ind] - Y_bar[mu + kn + ind]) * (Y[j + ind] - Y_bar[mu + 3*kn + ind]) * (Y[mu + (2*2 + 1)*kn + 0 + ind] - Y_bar[mu_pp + (2*2 + 1)*kn + kn + ind]) * (Y[mu + (2*2 + 1)*kn + j + ind] - Y_bar[mu_pp + (2*2 + 1)*kn + 3*kn + ind]))
        U_bar_2[j] = np.sum((Y[ind] - Y_bar[mu + kn + ind]) * (Y[j + ind] - Y_bar[mu + 3*kn + ind]) * (Y[mu + (2*2 + 1)*kn + 0 + ind] - Y_bar[mu + (2*2 + 1)*kn + kn + ind]) * (Y[mu + (2*2 + 1)*kn + 0 + ind] - Y_bar[mu + (2*2 + 1)*kn + 3*kn + ind]))
        
    S = np.zeros(J + 1)
    Bza = np.zeros(J + 1)
    S_1 = np.zeros(J + 1)
    Bza_1 = np.zeros(J + 1)
    sigma_r = np.ones(J)

    j = 0
    S[j+1] = (np.sum(UU_1[:,j+1]) + np.sum(UU_2[:,j+1]) - UU_1[0,j+1]) - (2*kn_p+1) * U_bar_1[j+1]
    Bza[j+1] = S[j+1] / N + U_bar_1[j+1] / N - R_est[j+1]**2

    S_1[j+1] = (np.sum(UU_3[:,j+1]) + np.sum(UU_4[:,j+1]) - UU_3[0,j+1]) - (2*kn_p+1) * U_bar_2[j+1]
    Bza_1[j+1] = S_1[j+1] / N + U_bar_2[j+1] / N - R_est[0] * R_est[j+1]

    for j in range(J):
        S[j+1] = (np.sum(UU_1[:,j+1]) + np.sum(UU_2[:,j+1]) - UU_1[0,j+1]) - (2*kn_p+1)*U_bar_1[j+1]
        Bza[j+1] = S[j+1]/N + U_bar_1[j+1]/N - R_est[j+1]**2

        S_1[j+1] = (np.sum(UU_3[:,j+1]) + np.sum(UU_4[:,j+1]) - UU_3[0,j+1]) - (2*kn_p+1)*U_bar_2[j+1]
        Bza_1[j+1] = S_1[j+1]/N + U_bar_2[j+1]/N - R_est[0]*R_est[j+1]
        sigma_r[j] = (R_est[0]**2*Bza[j+1] + R_est[j+1]**2*Bza[0] - 2*R_est[0]*R_est[j+1]*Bza_1[j+1]) / R_est[0]**4

    results = {}
    results['R'] = R_est/T
    results['Rstd'] = 1/np.sqrt(N/np.abs(Bza))/T
    results['rstd'] = 1/np.sqrt(N/np.abs(sigma_r))
    results['r'] = r_est
    results['Pvg'] = IV/T
    #results['Quarticity'] = Quarticity
    '''