import numpy as np
import pandas as pd
import pymc as mc
from scipy import signal
import statsmodels.api as sm
import matplotlib.pyplot as plt
# import seaborn as sn


#x 1) get it working with just one of the variables using an AR(1) model
#x 2) get it working with just one of the variables using an AR(2) model
# 3) get it working for the VAR(12) model
# 4) plot the projected (i.e., predicted) relative to the actual series
# 5) try some forecasting using the posterier mean

def load_data_arima():
    print 'Loading the data...\n'
    df = pd.read_csv('../../data sets/bayesian/NU_bayesian_time_series_ass1.tsv', sep = '\t')
    df.index = pd.to_datetime(df.index)
    y = df['log(CPI)']
    # print y
    return y

def mcmc_arima(y):
    print 'Beginning the fun stuff...\n'

    # defining priors
    ###### how does one decide on a prior?
    arima_precision = mc.Gamma('arima_precision', 2, 4)
    gamma1 = mc.Beta('gamma1', 2, 1)
    gamma2 = mc.Beta('gamma2', 2, 1)
    # gamma = mc.Uniform('gamma', -50, 50)
    # gamma = mc.Normal('gamma', 0, 1)

    # Instantiate the ARIMA model with our simulated data
    arima_mod = sm.tsa.SARIMAX(y, order=(2,1,0))

    # Create the stochastic (observed) component
    @mc.stochastic(dtype=sm.tsa.SARIMAX, observed=True)
    def arima(value=arima_mod, h=arima_precision, gamma1=gamma1, gamma2=gamma2):
        # Rejection sampling
        # if h < 0:
        # if gamma < -1 or gamma>1 or h < 0:
        # if gamma < 0 or h < 0:
        if gamma1 < 0 or gamma2 < 0 or h < 0: #I don't know what this is for...the following line never prints...maybe it is relevant if I choose priors that do not inherently constrain the draws
            print gamma1, gamma2, h
            return 0
        # print gamma1, gamma2, h #prints the draws
        return value.loglike(np.array([gamma1, gamma2, 1./h]), transformed=True)

    # Create the PyMC model
    arima_mc = mc.Model((arima_precision, gamma1, gamma2, arima)) # can these go in any order?

    # Create a PyMC sample
    arima_sampler = mc.MCMC(arima_mc)

    print 'Sampler action...\n'
    # Sample
    arima_sampler.sample(iter=10000, burn=1000, thin=10)

    # Plot traces
    mc.Matplot.plot(arima_sampler)
    plt.show()

    # # maximum likelihood
    arima_res = arima_mod.fit()
    print arima_res.summary()

def load_data_var():
    print 'Loading the data...\n'

    df = pd.read_csv('../../data sets/bayesian/NU_bayesian_time_series_ass1.tsv', sep = '\t')
    df.index = pd.to_datetime(df.index)
    df = df[['log(CPI)', 'log(IP)']]
    # df = df[['log(CPI)']]
    # df.drop(['day', 'month', 'year', 'log(M2)'], 1, inplace=True)
    print df.info()

    return df

def mcmc_var(y):
    print 'Beginning the fun stuff...\n'


#### when I get into estimating this bayesian style, I should start with just two variables instead of five.

    # # defining priors
    # ###### how does one decide on a prior?
    arima_precision1 = mc.Gamma('arima_precision1', 2, 4)
    arima_precision2 = mc.Gamma('arima_precision2', 2, 4)
    arima_precision12 = mc.Gamma('arima_precision12', 2, 4)
    b01 = mc.Normal('b01', 0, 0.0003)
    b11 = mc.Normal('b11', 0, 0.0003)
    b21 = mc.Normal('b21', 0, 0.0003)
    b02 = mc.Normal('b02', 0, 0.0003)
    b12 = mc.Normal('b12', 0, 0.0003)
    b22 = mc.Normal('b22', 0, 0.0003)
# ['const.log(CPI)', 'const.log(IP)', 'L1.log(CPI).log(CPI)', 'L1.log(IP).log(CPI)', 'L1.log(CPI).log(IP)', 'L1.log(IP).log(IP)', 'sqrt.var.log(CPI)', 'sqrt.cov.log(CPI).log(IP)', 'sqrt.var.log(IP)']

    # # Instantiate the ARIMA model with our simulated data
    varma_mod = sm.tsa.VARMAX(y, order=(1,0))
    print varma_mod.param_names

    # # Create the stochastic (observed) component
    @mc.stochastic(dtype=sm.tsa.VARMAX, observed=True)
    def varma(value=varma_mod, h1=arima_precision1, h2=arima_precision2, h12=arima_precision12, b01=b01, b11=b11, b21=b21, b02=b02, b12=b12, b22=b22):
    # def varma(value=varma_mod, h1=arima_precision1, b01=b01, b11=b11):
        # Rejection sampling
        # if h1 < 0:
        # if gamma < -1 or gamma>1 or h < 0:
        # if gamma < 0 or h < 0:
        # if gamma1 < 0 or gamma2 < 0 or h < 0: #I don't know what this is for...the following line never prints...maybe it is relevant if I choose priors that do not inherently constrain the draws
            # print gamma1, gamma2, h
            # return 0
        # print gamma1, gamma2, h #prints the draws
        # return value.loglike(np.array([b01, b11, b21, 1./h1]), np.array([b02, b12, b22, 2./h2]), transformed=True)
        return value.loglike(np.array([1./h1, 1./h2, 1./h12, b01, b11, b21, b02, b12, b22]), transformed=True)
        # return value.loglike(np.array([b01, b11, 1./h1]), transformed=True)

    # Create the PyMC model
    arima_mc = mc.Model((arima_precision1, arima_precision2, arima_precision12, b01, b11, b21, b02, b12, b22, varma)) # can these go in any order?
    # arima_mc = mc.Model((arima_precision1, b01, b11, varma)) # can these go in any order?

    # Create a PyMC sample
    arima_sampler = mc.MCMC(arima_mc)

    print 'Sampler action...\n'
    # Sample
    arima_sampler.sample(iter=10000, burn=1000, thin=10)

    # Plot traces
    mc.Matplot.plot(arima_sampler)
    plt.show()

    # # # maximum likelihood
    varma_res = varma_mod.fit()
    print varma_res.summary()


if __name__=="__main__":
    # y = load_data_arima()
    # mcmc_arima(y)
    y = load_data_var()
    mcmc_var(y)
