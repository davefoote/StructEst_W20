import numpy as np
import pandas as pd
import scipy.stats as sts
from matplotlib import pyplot as plt
import requests
import scipy.optimize as opt
import scipy.special as spc
import math

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
                                    edgecolor='k', density=True)
    else:
        count, bins, ignored = plt.hist(data, num_bins, density=True,
                                    edgecolor='k')
    print(count.max())
    plt.title(title, fontsize=20)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(xlim)
    
    return count
    
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
        
    if len(params) == 4:
        p0 = params[0]
        p1 = params[1]
        p2 = params[2]
        p3 = params[3]

        return np.log(f(x, p0, p1, p2, p3)).sum()

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

def mle4(guess, f, data):
    p1_i = guess[0]
    p2_i = guess[1]
    p3_i = guess[2]
    p4_i = guess[3]
    p_init = np.array([p1_i, p2_i, p3_i, p4_i])
    data[data < 1e-10] = 1e-10
    results_uncstr = opt.minimize(f, p_init, args=(data))
    p1, p2, p3, p4 = results_uncstr.x
    print(results_uncstr)
    
    return p1, p2, p3, p4

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
    x[x < 1e-10] = 1e-10
    rv = ((1 / ((beta ** alpha) * spc.gamma(alpha))) *
                (x ** (alpha - 1)) * (np.exp(-x/beta)))
    

    return rv

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
    print(data.shape)
    print(params)
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
        --------------------------------------------------------------------
    Returns the PDF values from the four-parameter generalized beta 2
    (GB2) distribution. See McDonald and Xu (1995).

    (GB2): f(x; a, b, p, q) = (a * (x ** ((a*p) - 1))) /
        ((b ** (a * p)) * spc.beta(p, q) *
        ((1 + ((x / b) ** a)) ** (p + q)))
    x in [0, infty), alpha, beta, m > 0
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, values in the support of generalized beta 2
            (GB2) distribution
    params = tuple, containing the following:
    aa    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    bb    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    pp    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    qq    = scalar > 0, generalized beta 2 (GB2) distribution parameter

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        spc.beta()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, pdf values from generalized beta 2 (GB2)
               distribution corresponding to xvals given parameters aa,
               bb, pp, and qq

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    aa = params[0]
    bb = params[1]
    pp = params[2]
    qq = params[3]
    x[x < 1e-10] = 1e-10

    return np.float64((aa * (x ** (aa * pp - 1))) / np.exp((spc.logsumexp(bb) * (spc.logsumexp(aa * pp)
                                                    + spc.logsumexp(spc.beta(pp, qq)))
                                                    + spc.logsumexp(1 + ((x / bb) ** aa) ** 
                                                       (pp + qq)))))


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

def crit(func, params, data):
    return -log_lik(data, params, func)

'''
from prof
'''
def GB2_pdf(xvals, aa, bb, pp, qq):
    '''
    --------------------------------------------------------------------
    Returns the PDF values from the four-parameter generalized beta 2
    (GB2) distribution. See McDonald and Xu (1995).

    (GB2): f(x; a, b, p, q) = (a * (x ** ((a*p) - 1))) /
        ((b ** (a * p)) * spc.beta(p, q) *
        ((1 + ((x / b) ** a)) ** (p + q)))
    x in [0, infty), alpha, beta, m > 0
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, values in the support of generalized beta 2
            (GB2) distribution
    aa    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    bb    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    pp    = scalar > 0, generalized beta 2 (GB2) distribution parameter
    qq    = scalar > 0, generalized beta 2 (GB2) distribution parameter

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        spc.beta()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, pdf values from generalized beta 2 (GB2)
               distribution corresponding to xvals given parameters aa,
               bb, pp, and qq

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = \
        np.float64((aa * (xvals ** (aa * pp - 1))) / ((bb ** (aa * pp)) *
                   spc.beta(pp, qq) *
                   ((1 + ((xvals / bb) ** aa)) ** (pp + qq))))

    return pdf_vals