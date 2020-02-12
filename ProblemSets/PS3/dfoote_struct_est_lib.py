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

def mle4(guess, f, data):
    p_init = np.array(list(guess))
    results_uncstr = opt.minimize(f, p_init, args=(data))
    p1, p2, p3 = results.uncstr.x
    
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

    pdf_vals = \
        np.float64((aa * (xvals ** (aa * pp - 1))) / ((bb ** (aa * pp)) *
                   spc.beta(pp, qq) *
                   ((1 + ((xvals / bb) ** aa)) ** (pp + qq))))

    return pdf_vals


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
GMM
'''
def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), variance(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of test scores data
    var_data  = scalar > 0, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    var_data = xvals.var()
    
    return mean_data, var_data


def model_moments(mu, sigma, pdf_func):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), variance(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cut_lb = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar lower bound value of distribution. Values below
             this value have zero probability
    cut_ub = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    var_model  = scalar > 0, variance of test scores from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, var_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * pdf_func(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, cut_lb, cut_ub)
    x2fx = lambda x: ((x - mean_model) ** 2) * trunc_norm_pdf(x, mu, sigma, cut_lb, cut_ub) 
    (var_model, v_m_err) = intgr.quad(x2fx, cut_lb, cut_ub)
    
    return mean_model, var_model

def model_moments_ln(mu, sigma, bin_bound_list):
    bins = len(bin_bound_list) - 1
    model_moments = np.zeros(bins)
    for bin_ind in range(bins):
        if bin_ind == 0:
            model_moments[bin_ind] = \
                sts.lognorm.cdf(bin_bound_list[bin_ind], s=sigma,
                                scale=np.exp(mu))
        elif bin_ind > 0 and bin_ind < bins - 1:
            model_moments[bin_ind] = \
                (sts.lognorm.cdf(bin_bound_list[bin_ind], s=sigma,
                                 scale=np.exp(mu)) -
                 sts.lognorm.cdf(bin_bound_list[bin_ind - 1], s=sigma,
                                 scale=np.exp(mu)))
        elif bin_ind == bins - 1:
            model_moments[bin_ind] = \
                (1 - sts.lognorm.cdf(bin_bound_list[bin_ind - 1],
                                     s=sigma, scale=np.exp(mu)))

    return model_moments


def err_vec(xvals, mu, sigma, bins, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    bins = the bins from assignment 3
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        model_moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, var_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_model, var_model = model_moments(mu, sigma, bins)
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cut_lb = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar lower bound value of distribution. Values below
             this value have zero probability
    cut_ub = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cut_lb, cut_ub, W = args
    err = err_vec(xvals, mu, sigma, cut_lb, cut_ub, simple=False)
    crit_val = err.T @ W @ err
    
    return crit_val