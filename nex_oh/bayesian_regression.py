import pymc
import pymc3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#
# def linear_setup(df, ind_cols, dep_col):
#     '''
#     Inputs: pandas Data Frame, list of strings for the independent variables,
#     single string for the dependent variable
#     Output: PyMC Model
#     '''
#
#     # model our intercept and error term as above
#     b0 = pymc.Normal('b0', 0, 0.0001)
#     err = pymc.Uniform('err', 0, 500)
#
#     # initialize a NumPy array to hold our betas
#     # and our observed x values
#     b = np.empty(len(ind_cols), dtype=object)
#     x = np.empty(len(ind_cols), dtype=object)
#
#     # loop through b, and make our ith beta
#     # a normal random variable, as in the single variable case
#     for i in range(len(b)):
#         b[i] = pymc.Normal('b' + str(i + 1), 0, 0.0001)
#
#     # loop through x, and inform our model about the observed
#     # x values that correspond to the ith position
#     for i, col in enumerate(ind_cols):
#         x[i] = pymc.Normal('b' + str(i + 1), 0, 1, value=np.array(df[col]), observed=True)
#
#     # as above, but use .dot() for 2D array (i.e., matrix) multiplication
#     @pymc.deterministic
#     def y_pred(b0=b0, b=b, x=x):
#         return b0 + b.dot(x)
#
#     # finally, "model" our observed y values as above
#     y = pymc.Normal('y', y_pred, err, value=np.array(df[dep_col]), observed=True)
#
#     return pymc.Model([b0, pymc.Container(b), err, pymc.Container(x), y, y_pred])
#
# test_model = linear_setup(float_df, ['weight', 'acceleration'], 'mpg')
# mcmc = pymc.MCMC(test_model)
# mcmc.sample(100000, 20000)


def linear_model_from_scratch(df):
    b0 = pymc.Normal('b0', 0, 0.0003)
    b1 = pymc.Normal('b1', 0, 0.0003)

    err = pymc.Uniform('err', 0, 500)

    x_weight = pymc.Normal('weight', 0, 1, value=np.array(df['X']), observed=True)

    @pymc.deterministic
    def pred(b0=b0, b1=b1, x=x_weight):
        return b0 + b1*x

    y = pymc.Normal('y', pred, err, value=np.array(df['y']), observed=True)

    model = pymc.Model([pred, b0, b1, y, err, x_weight])
    mcmc = pymc.MCMC(model)
    mcmc.sample(50000, 20000)

    print np.mean(mcmc.trace('b1')[:])
    print stats.mode(mcmc.trace('b1')[:])

    fig = plt.figure(figsize = (13,10))
    ax = fig.add_subplot(1,1,1)
    ax.hist(mcmc.trace('b1')[:], bins=50)
    plt.show()


def linear_model_from_scratch3(df):
    with pymc3.Model() as model:
        pymc3.glm.glm('y ~ X', df)
        start = pymc3.find_MAP()
        step = pymc3.NUTS(scaling=start)
        trace = pymc3.sample(2000, step, progressbar=True)

    print 'plotting the shiz now'
    fig = plt.figure(figsize=(7, 7))
    pymc3.traceplot(trace[100:])
    # plt.tight_layout()
    plt.show()



def linear_model_baked(df):
    b0 = pymc.Normal('b0', 0, 0.0003)
    b1 = pymc.Normal('b1', 0, 0.0003)

    err = pymc.Uniform('err', 0, 500)

    x_weight = pymc.Normal('weight', 0, 1, value=np.array(df['X']), observed=True)

    @pymc.deterministic
    def pred(b0=b0, b1=b1, x=x_weight):
        return b0 + b1*x

    y = pymc.Normal('y', pred, err, value=np.array(df['y']), observed=True)

    model = pymc.Model([pred, b0, b1, y, err, x_weight])
    mcmc = pymc.MCMC(model)
    mcmc.sample(50000, 20000)

    print np.mean(mcmc.trace('b1')[:])
    print stats.mode(mcmc.trace('b1')[:])

    fig = plt.figure(figsize = (13,10))
    ax = fig.add_subplot(1,1,1)
    ax.hist(mcmc.trace('b1')[:], bins=50)
    plt.show()

if __name__=="__main__":
    df = pd.DataFrame(np.random.uniform(0, 10, 1000), columns=['X'])
    df['y'] = 2*df['X'] + np.random.normal(0, 1, 1000)

    # fig = plt.figure(figsize=(13, 10))
    # ax = fig.add_subplot(1,1,1)
    # ax.scatter(df['X'], df['y'])
    # plt.show()

    # linear_model_from_scratch(df)
    linear_model_from_scratch3(df)
    # linear_model_baked(df)




    # df = pd.read_csv('../../data sets/bayesian/NU_bayesian_time_series_ass1.tsv', sep = '\t')
    # linear_setup(df, ind_cols, dep_col)
