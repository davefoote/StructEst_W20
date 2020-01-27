import numpy as np
import pandas as pd
import scipy.stats as sts
from matplotlib import pyplot as plt
import requests
import scipy.optimize as opt

'''
Dave Foote's Structural Estimation Library
'''

'''
exploratory data analysis
'''
def summarize_column(array):
    '''
    given a numpy array this function will produce and return a mean,
    max, min, std of the numbers in the series and produce a simple histogram
    '''
    mean = array.mean()
    median = np.median(array)
    maximum = array.max()
    minimum = array.min()
    std = array.std()
    
    rd = {'mean':mean, 'med':median, 'max':maximum, 'min':minimum, 'std':std}
    
    for k, v in rd.items():
        print(k, ' : ', v)

    return rd

def histogram(data, num_bins, title, xlab, ylab, xlim, og_data=np.array([])):
    '''
    generate a plt histogram of a np array
    '''
    if og_data.any():
        wt = (1/og_data.shape[0]) * np.ones_like(data)
        count, bins, ignored = plt.hist(data, num_bins,
                                    edgecolor='k', weights=wt)
    else:
        count, bins, ignored = plt.hist(data, num_bins, density=True,
                                    edgecolor='k')
    print(count.max())
    plt.title(title, fontsize=20)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(xlim)
    
'''
mle functions
'''
def log_lik(x, params, f):
    '''
    Compute the log likelihood function for data xvalsgiven given distribution
    that uses 2 parameters and those 2 paramters
    '''
    if len(params) == 2:
        p0 = params[0]
        p1 = params[1]

        return np.log(f(x, p0, p1)).sum()
    if len(params) == 3:
        p0 = params[0]
        p1 = params[1]
        p2 = params[2]

        return np.log(f(x, p0, p1, p2)).sum()

def mle(guess, f, data):
    '''
    runs MLE to solve for alpha and beta that best fit a gamma dist to given
    data
    
    FOR 2 PARAMETER PROBLEMS
    
    inputs:
        guess: an initial guess, probably provided by a _firstguess function
        f: the function to be used to generate pdfs for the mle estimation
        data: np array of data
    '''
    alpha_init = guess[0]
    beta_init = guess[1]
    p_init = np.array([alpha_init, beta_init])
    results_uncstr = opt.minimize(f, p_init, args=(data))
    p1, p2 = results_uncstr.x
    
    return p1, p2

def mle3(guess, f, data):
    p1_init = guess[0]
    p2_init = guess[1]
    p3_init = guess[2]
    p_init = np.array([p1_init, p2_init, p3_init])
    results_uncstr = opt.minimize(f, p_init, args=(data))
    p1, p2, p3 = results_uncstr.x
    
    return p1, p2, p3

'''
gamma functions
'''
    
def gamma_firstguess(dic):
    '''
    takes output of summary and generates a first guess for
    alpha and beta of a gamma distribution over the data used to
    generate the summary
    '''

    rate = dic['std']**2 / dic['mean']
    shape = dic['mean'] /rate

    print('SHAPE: ', shape)
    print('RATE: ', rate)

    return (shape, rate)

def make_gamma_pdf(x, alpha, beta):
    '''
    generates pdfs values from a gamma distribution fitting specifiedx, alpha,
    and beta
    '''
    vals = sts.gamma.pdf(x, alpha, scale=beta)
    vals[vals < 1e-10] = 1e-10

    return vals

def gamma_crit(params, data):
    '''
    This function generates the negative of the gamma log likelihood function
    given parameters and data
    ___________________________________________________________________________
    ---------------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([alpha, beta])
    alpha = scalar, shape of gamma distributed random variable
    beta = scalar, rate of gamma distributed random variable
    pdf_function = the function you would like to use for pdf generating
    data = (N,) vector, values of the gamma distributed random variable
    '''
    log_lik_value = log_lik(data, params, make_gamma_pdf)

    return -log_lik_value

'''
gen gamma functions
'''
def gengamma_firstguess(dic, data):
    '''
    takes output of summary and generates a first guess for
    alpha and beta of a gamma distribution over the data used to
    generate the summary
    '''
    gamma_init_params = gamma_firstguess(dic)
    
    alpha_mle, beta_mle = mle(gamma_init_params, gamma_crit, data)

    return (alpha_mle, beta_mle, 1)

def make_gengamma_pdf(x, alpha, beta, m):
    '''
    generates pdfs values from a gamma distribution fitting specifiedx, alpha,
    and beta
    '''
    vals = sts.gengamma.pdf(x, alpha, m, scale=beta)
    vals[vals < 1e-10] = 1e-10

    return vals

def gengamma_crit(params, data):
    '''
    This function generates the negative of the gamma log likelihood function
    given parameters and data
    ___________________________________________________________________________
    ---------------------------------------------------------------------------
    INPUTS:
    params = (3,) vector, ([alpha, beta, m])
    alpha = scalar, shape of gamma distributed random variable
    beta = scalar, rate of gamma distributed random variable
    m = scalar, m of gamma distributed random variable
    pdf_function = the function you would like to use for pdf generating
    data = (N,) vector, values of the gamma distributed random variable
    '''
    log_lik_value = log_lik(data, params, make_gengamma_pdf)

    return -log_lik_value

'''
GB2 Functions
'''
def gb2_firstguess(gamma_output):
    alpha = gamma_output[0]
    beta = gamma_output[1]
    m = gamma_output[2]
    q = gamma_output[3]
    
    b = q**(1 / m)
    p = alpha / m
    
    return (m, b, p, q)

def make_gb2_pdf(x, params):
    '''
    generates pdfs values from a gamma distribution fitting specifiedx, alpha,
    and beta
    '''
    a = params[0]
    b = params[1]
    p = params[2]
    q = params[3]
    vals = sts.beta.pdf()
    vals[vals < 1e-10] = 1e-10

    return vals

def gb2_crit(params, data):
    '''
    This function generates the negative of the gamma log likelihood function
    given parameters and data
    ___________________________________________________________________________
    ---------------------------------------------------------------------------
    INPUTS:
    params = (4,) vector, ([alpha, beta, p, q])
    alpha = scalar, shape of gamma distributed random variable
    beta = scalar, rate of gamma distributed random variable
    p = scalar
    q = scalar
    pdf_function = the function you would like to use for pdf generating
    data = (N,) vector, values of the gamma distributed random variable
    '''
    log_lik_value = log_lik(data, params, make_gb2_pdf)

    return -log_lik_value
