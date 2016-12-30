import numpy as np
import pandas as pd
import pymc as mc
from scipy import signal
import statsmodels.api as sm
import matplotlib.pyplot as plt
# import seaborn as sn

y = pd.read_csv('../job assessment and hackerrank data/nexosis skill-assessment/DSTakeHome/ts4.csv')

y.date = pd.to_datetime(y.date)
y.index = y.date
y.drop(['date'], 1, inplace=True)

# df_ts4_test = pd.read_csv('../hackerrank_data/skill-assessment/DSTakeHome/ts4_test.csv')
# df_ts4_test.date = pd.to_datetime(df_ts4_test.date)
# df_ts4_test.index = df_ts4_test.date
# df_ts4_test.drop(['date'], 1, inplace=True)





# T = 1000
# sigma2_eps0 = 3
# sigma2_eta0 = 10
#
# # Simulate data
# np.random.seed(1234)
# eps = np.random.normal(scale=sigma2_eps0**0.5, size=T)
# eta = np.random.normal(scale=sigma2_eta0**0.5, size=T)
# mu = np.cumsum(eta)
# y = mu + eps

# # Plot the time series
# fig = plt.figure(figsize = (13,10))
# ax = fig.add_subplot(1,1,1)
# ax.plot(y)
# ax.set(xlabel='$T$', title='Simulated series')
# plt.show()



###### figure out how to do a

arima_precision = mc.Gamma('arima_precision', 2, 4)
gamma = mc.Beta('gamma', 2, 1)

# Instantiate the ARIMA model with our simulated data
arima_mod = sm.tsa.SARIMAX(y, order=(0,1,1))

# Create the stochastic (observed) component
@mc.stochastic(dtype=sm.tsa.SARIMAX, observed=True)
def arima(value=arima_mod, h=arima_precision, gamma=gamma):
    # Rejection sampling
    if gamma < 0 or h < 0:
        return 0
    return value.loglike(np.r_[-gamma, 1./h], transformed=True)

# Create the PyMC model
arima_mc = mc.Model((arima_precision, gamma, arima))

# Create a PyMC sample
arima_sampler = mc.MCMC(arima_mc)

# Sample
arima_sampler.sample(iter=10000, burn=1000, thin=10)

# Plot traces
mc.Matplot.plot(arima_sampler)
plt.show()

# # maximum likelihood
arima_res = arima_mod.fit()
print arima_res.summary()
#
#
# #forecasting
# fig, ax = plt.subplots(figsize=(13,3))
#
# ax.plot(y, label='y')
# ax.plot(ll_res.predict()[0], label='Local level')
# ax.plot(arima_res.predict()[0], label='ARIMA(0,1,1)')
# ax.legend(loc='lower right');








# # Priors
# precision = mc.Gamma('precision', 2, 4)
# ratio = mc.Gamma('ratio', 2, 1)
#
# # Likelihood calculated using the state-space model
# class LocalLevel(sm.tsa.statespace.MLEModel):
#     def __init__(self, endog):
#         # Initialize the state space model
#         super(LocalLevel, self).__init__(endog, k_states=1,
#                                          initialization='approximate_diffuse',
#                                          loglikelihood_burn=1)
#
#         # Initialize known components of the state space matrices
#         self.ssm['design', :] = 1
#         self.ssm['transition', :] = 1
#         self.ssm['selection', :] = 1
#
#     @property
#     def start_params(self):
#         return [1. / np.var(self.endog), 1.]
#
#     @property
#     def param_names(self):
#         return ['h_inv', 'q']
#
#     def update(self, params, transformed=True, **kwargs):
#         params = super(LocalLevel, self).update(params, transformed, **kwargs)
#
#         h, q = params
#         sigma2_eps = 1. / h
#         sigma2_eta = q * sigma2_eps
#
#         self.ssm['obs_cov', 0, 0] = sigma2_eps
#         self.ssm['state_cov', 0, 0] = sigma2_eta
#
# # Instantiate the local level model with our simulated data
# ll_mod = LocalLevel(y)
# ll_mod.filter(ll_mod.start_params)
#
# # Create the stochastic (observed) component
# @mc.stochastic(dtype=LocalLevel, observed=True)
# def local_level(value=ll_mod, h=precision, q=ratio):
#     return value.loglike([h, q], transformed=True)
#
# # Create the PyMC model
# ll_mc = mc.Model((precision, ratio, local_level))
#
# # Create a PyMC sample
# ll_sampler = mc.MCMC(ll_mc)
#
