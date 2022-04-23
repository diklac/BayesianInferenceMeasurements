# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:25:22 2022

@author: dikla
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import arviz as az
import pymc3 as pm
from pymc3.distributions import Interpolated
from tqdm import tqdm

# Interface
n = 7 # number of measurements
alpha0 = 25.1 # intercept
beta0 = 3.2 # slope
sigma_exp = 2.0 # std of measurements about the true value
xi = 1.5 # lower part of measurement points interval
xf = 10.5 # upper part of measurement points interval
num_repeat = 2000 # when testing the results, how many times to repeat
confidence_level = 0.95 # for confidence intervals
line_width = 3
errorbar_capsize = 3
errorbar_capthick = 3
num_pts_factor = 50

# funcs
def calc_aux(x, ys):
    xbar = np.mean(x)
    ybar = np.mean(ys)
    Sxy = np.dot(x - xbar, ys - ybar)
    Sxx = np.dot(x - xbar, x - xbar)
    return (xbar, ybar, Sxx, Sxy)
    
def calc_point_estimation(x, ys):
    '''
    Calculate the point estimation of intercept and slope
    using least squares.

    Parameters
    ----------
    x : independent variable
    ys : measurements

    Returns
    -------
    alpha_est : least squares estimate of intercept
    beta_est : least squares estimate of slope
    '''
    (xbar, ybar, Sxx, Sxy) = calc_aux(x, ys)
    beta_est = Sxy/Sxx
    alpha_est = ybar - beta_est*xbar
    return (alpha_est, beta_est)

def calc_estimation_variance(x, ys, sigma_exp, n):
    '''
    Calculate the estimation variance (of the BLUE) of 
    intercept and slope.

    Parameters
    ----------
    x : independent variable
    ys : measurements
    sigma_exp : known measurements error
    n : number of measurements

    Returns
    -------
    var_alpha_est : estimation variance of BLUE of intercept
    var_beta_est : estimation variance of BLUE of slope
    '''
    (xbar, ybar, Sxx, Sxy) = calc_aux(x, ys)
    var_beta_est = sigma_exp**2/Sxx
    var_alpha_est = np.dot(x, x)*sigma_exp**2/(n*Sxx) 
    return (var_alpha_est, var_beta_est)

def calc_CI_with_sigma(x, ys, sigma_exp, confidence_level):
    '''
    Calculate the symmetric CI in confidence_level for intercept
    and slope, assuming normally distributed measurements ys with
    known std sigma_exp. The CI is based on the normal distribution
    of the unbiased estimators of intercept and slope.

    Parameters
    ----------
    x : independent variable
    ys : measurements
    sigma_exp : measurements std
    confidence_level : in [0,1]

    Returns
    -------
    alpha_CI_half_sigma : half of the symmetric CI (about mean) of intercept
    beta_CI_half_sigma : half of the symmetric CI (about mean) of slope
    '''
    (xbar, ybar, Sxx, Sxy) = calc_aux(x, ys)
    z_star = stats.norm.ppf(1 - 0.5*(1 - confidence_level))
    alpha_CI_half_sigma = z_star*np.sqrt(sigma_exp**2*np.dot(x, x)/(n*Sxx))
    beta_CI_half_sigma = z_star*np.sqrt(sigma_exp**2/Sxx)
    return (alpha_CI_half_sigma, beta_CI_half_sigma)

def calc_CI_without_sigma(x, ys, n, confidence_level):
    '''
    Calculate the symmetric CI in confidence_level for intercept
    and slope, assuming normally distributed measurements ys with
    unknown std and n measurements. The CI is based on the Student's
    t distribution of the ratio of unbiased estimate of parameter and
    the unbiased estimate of the variance.

    Parameters
    ----------
    x : independent variable
    ys : measurements
    n : number of measurements
    confidence_level : in [0,1]

    Returns
    -------
    alpha_CI_half_sigma : half of the symmetric CI (about mean) of intercept
    beta_CI_half_sigma : half of the symmetric CI (about mean) of slope
    '''
    (alpha_est, beta_est) = calc_point_estimation(x, ys)
    (xbar, ybar, Sxx, Sxy) = calc_aux(x, ys)
    S2 = (1/(n - 2))*np.dot(ys - alpha_est - beta_est*x, ys - alpha_est - beta_est*x)
    t_n_2 = stats.t.ppf(1 - 0.5*(1 - confidence_level), n - 2)
    alpha_CI_half = t_n_2*np.sqrt(S2*np.dot(x, x)/(n*Sxx))
    beta_CI_half = t_n_2*np.sqrt(S2/Sxx)
    return (alpha_CI_half, beta_CI_half)

def calc_fit_CI_with_sigma(x, ys, n, sigma_exp, confidence_level):
    '''
    Calculate the fit CI when knowning sigma_exp.

    Parameters
    ----------
    x : independent variable
    ys : measurements
    n : number of measurements
    sigma_exp : measurements std
    confidence_level : in [0,1]
    '''
    (xbar, ybar, Sxx, Sxy) = calc_aux(x, ys)
    z_star = stats.norm.ppf(1 - 0.5*(1 - confidence_level))
    half_interval = z_star*sigma_exp*np.sqrt(1.0/n + (x - xbar)**2/Sxx)
    return half_interval

def calc_fit_CI_without_sigma(x, ys, n, confidence_level):
    '''
    Calculate the fit CI when knowning sigma_exp.

    Parameters
    ----------
    x : independent variable
    ys : measurements
    n : number of measurements
    sigma_exp : measurements std
    confidence_level : in [0,1]
    '''
    (alpha_est, beta_est) = calc_point_estimation(x, ys)
    (xbar, ybar, Sxx, Sxy) = calc_aux(x, ys)
    S2 = (1/(n - 2))*np.dot(ys - alpha_est - beta_est*x, ys - alpha_est - beta_est*x)
    t_n_2 = stats.t.ppf(1 - 0.5*(1 - confidence_level), n - 2)
    half_interval = t_n_2*np.sqrt(S2*(1.0/n + (x - xbar)**2/Sxx))
    return half_interval


# set plotting style
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.autolayout' : True})
plt.tight_layout()

# set rng
rng = np.random.default_rng()

# plot large and small error bars
ax1 = plt.subplot(1, 2, 1)
plt.errorbar(['super accurate phys const'], [137], [20], linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.title('Me')
plt.subplot(1, 2, 2, sharey = ax1)
plt.errorbar(['super accurate phys const'], [137], [7], linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.title('Other Lab')

# plot partially negative interval for nonnegative value
plt.figure()
plt.errorbar(['cookie mass'], [20], [35], linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.ylabel('mass [g]')
plt.title('Nonsensical Error Bar')

# generate the measurements from a normal model
x = np.linspace(xi, xf, n)
ys_clean = alpha0 + beta0*x
ys = rng.normal(ys_clean, sigma_exp)

# plot the noisy measurements
plt.figure()
plt.errorbar(x, ys, sigma_exp, linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.xlabel(r'$\Delta x$ [mm]')
plt.ylabel(r'$I_{absorbed}$ $[W/cm^2]$')

# without using any statistical assumptions, calculate the fit
# parameters based on least squares
(alpha_est, beta_est) = calc_point_estimation(x, ys)
print('-----------Estimates-----------------')
print('alpha = {0:.2f}, beta = {1:.2f}'.format(alpha_est, beta_est))
y_fit = alpha_est + beta_est*x

# plot the fit
plt.figure()
plt.errorbar(x, ys, sigma_exp, linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.plot(x, y_fit, linewidth = line_width)
plt.xlabel(r'$\Delta x$ [mm]')
plt.ylabel(r'$I_{absorbed}$ $[W/cm^2]$')

# using some statistical assumptions, namely that
# the measurements have mean \alpha + \beta*x_i and variance
# \sigma_{exp}^2, the estimators' variance can be found
(var_alpha_est, var_beta_est) = calc_estimation_variance(x, ys, sigma_exp, n)
print('Var[alpha] = {0:.2f}, Var[beta] = {1:.2f}'.format(var_alpha_est, var_beta_est))

# using a statistical model of the measurements, this time with
# a distribution (normal), the 95% CIs can be found.
# If we know sigma_exp, a realization of the CI
(alpha_CI_half_sigma, beta_CI_half_sigma) = calc_CI_with_sigma(x, ys, sigma_exp, confidence_level)
print('CI[alpha|sigma_exp] = [{0:.2f}, {1:.2f}], CI[beta|sigma_exp] = [{2:.2f}, {3:.2f}]'.format(alpha_est - alpha_CI_half_sigma, alpha_est + alpha_CI_half_sigma, beta_est - beta_CI_half_sigma, beta_est + beta_CI_half_sigma))

# If we don't know sigma_exp, a realization of the CI
(alpha_CI_half, beta_CI_half) = calc_CI_without_sigma(x, ys, n, confidence_level)
print('CI[alpha] = [{0:.2f}, {1:.2f}], CI[beta] = [{2:.2f}, {3:.2f}]'.format(alpha_est - alpha_CI_half, alpha_est + alpha_CI_half, beta_est - beta_CI_half, beta_est + beta_CI_half))

# plot the various estimation error measures for alpha, beta
plt.figure()
ax = np.array([1, 2, 3])
ax = ['Var', 'CI sig', 'CI']
plt.subplot(1, 2, 1)
plt.errorbar(ax, [alpha_est for item in ax], [var_alpha_est, alpha_CI_half_sigma, alpha_CI_half], linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.title(r'$\hat{\alpha}$')
plt.subplot(1, 2, 2)
plt.errorbar(ax, [beta_est for item in ax], [var_beta_est, beta_CI_half_sigma, beta_CI_half], linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.title(r'$\hat{\beta}$')

# calculate fit CIs
CI_fit_sigma = calc_fit_CI_with_sigma(x, ys, n, sigma_exp, confidence_level)
CI_fit_without_sigma = calc_fit_CI_without_sigma(x, ys, n, confidence_level)

# plot the fit with CIs
fig, ax = plt.subplots()
ax.errorbar(x, ys, sigma_exp, linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
ax.plot(x, y_fit, linewidth = line_width)
ax.fill_between(x, y_fit + CI_fit_sigma, y_fit - CI_fit_sigma, alpha = 0.3)
ax.fill_between(x, y_fit + CI_fit_without_sigma, y_fit - CI_fit_without_sigma, alpha = 0.3)
plt.xlabel(r'$\Delta x$ [mm]')
plt.ylabel(r'$I_{absorbed}$ $[W/cm^2]$')

# plot the scalar factor in the CIs for knowing sigma_exp (normal)
# and not knowing sigma_exp (Student's t)
qs = np.linspace(0.9, 1, num_pts_factor)
fac_norm = stats.norm.ppf(qs)
fac_t = stats.t.ppf(qs, n - 2)
fac_t_10 = stats.t.ppf(qs, 10*n - 2)
fac_t_half = stats.t.ppf(qs, np.ceil(0.5*n) - 2)
plt.figure()
qs_ax = 2*qs - 1
plt.plot(qs_ax, fac_norm, '-', linewidth = line_width)
plt.plot(qs_ax, fac_t_10, '--', linewidth = line_width)
plt.plot(qs_ax, fac_t, '-', linewidth = line_width)
plt.plot(qs_ax, fac_t_half, '-.', linewidth = line_width)
plt.xlabel('confidence level')
plt.ylabel('factor')
plt.legend(['normal', r'$t_{10n-2}$', r'$t_{n-2}$', r'$t_{\frac{1}{2}n-2}$'])

# test the results
estim = np.zeros((num_repeat, 2))
CIs_sigma = np.zeros((num_repeat, 2))
CIs = np.zeros((num_repeat, 2))
print('-----------Statistics-----------------')
for ii in tqdm(range(num_repeat)):
    # measurements realizations and estimates
    ys_r = rng.normal(alpha0 + beta0*x, sigma_exp)
    estimates = np.array(calc_point_estimation(x, ys_r))
    estim[ii, :] = estimates
    
    # CI knowing sigma
    half_int_sigma = np.array(calc_CI_with_sigma(x, ys_r, sigma_exp, confidence_level))
    intervals_sigma_L = estimates - half_int_sigma
    intervals_sigma_U = estimates + half_int_sigma
    CIs_sigma[ii, :] = np.logical_and((alpha0, beta0) >= intervals_sigma_L, (alpha0, beta0) <= intervals_sigma_U)
    
    # CI not knowing sigma
    half_int = np.array(calc_CI_without_sigma(x, ys_r, n, confidence_level))
    intervals_L = estimates - half_int
    intervals_U = estimates + half_int
    CIs[ii, :] = np.logical_and((alpha0, beta0) >= intervals_L, (alpha0, beta0) <= intervals_U)
    
coverage_sigma = CIs_sigma.sum(0)/num_repeat
coverage_without_sigma = CIs.sum(0)/num_repeat
estim_var = np.var(estim, 0)
print('\nCalculated estimation variance ({4:.2f}, {5:.2f})\nCI coverage with sigma: ({0:.2f}, {1:.2f})\nCI coverage without sigma: ({2:0.2f}, {3:.2f})'.format(coverage_sigma[0], coverage_sigma[1], coverage_without_sigma[0], coverage_without_sigma[1], estim_var[0], estim_var[1]))
    

    
    

