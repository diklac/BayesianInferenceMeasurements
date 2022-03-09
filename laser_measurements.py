# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:31:17 2022

@author: dikla
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import cm
import arviz as az
import pymc3 as pm


plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.autolayout' : True})
plt.tight_layout()

def Iout(z, Iin, g):
    return Iin*(1 - g*z)


num_measurements = 10
sigma_exp = 2.0
Iin = 25.0
g = 0.13
delta_z = np.linspace(1.1, 5.7, num_measurements)
line_width = 3
errorbar_capsize = 3
errorbar_capthick = 3
num_hist = 10000

# prior
Iin_0 = 15.0
sigma_Iin_0 = 15
g_0 = 1.0
sigma_g_0 = 2.0

# MC
num_samples = 50000

    

# calculate the "real" measurements
ys_clean = Iin*(1 - g*delta_z)

# calculate noisy measurements
rng = np.random.default_rng()
ys = rng.normal(ys_clean, sigma_exp)

# fit using scipy (nonlinear least squares)
fit_func = lambda x, a, b: a - b*x
popt, pcov = curve_fit(fit_func, delta_z, ys, sigma = sigma_exp*np.ones(ys.size), absolute_sigma = True)
perr = np.sqrt(np.diag(pcov))
g_fit_scipy = popt[1]/popt[0]
sigma_g_fit_scipy = g_fit_scipy*np.sqrt(perr[0]**2/popt[0]**2 + perr[1]**2/popt[1]**2)
print('Fit using Scipy: a = %.2f (%.2f), b = %.2f (%.2f), g = %.2f (%.2f)' % (popt[0], perr[0], popt[1], perr[1], g_fit_scipy, sigma_g_fit_scipy))

# fit by hand, minimum sum of squared residuals
y_bar = np.mean(ys)
z_bar = np.mean(delta_z)
yz_bar = np.mean(np.dot(delta_z, ys))
z2_bar = np.mean(np.dot(delta_z, delta_z))
b_star = (1.0/(z2_bar - z_bar**2))*(y_bar*z_bar - yz_bar)
a_star = y_bar - b_star*z_bar
# these are the estimators stds
sigma_b = sigma_exp/np.sqrt(num_measurements*(z2_bar - z_bar**2))
sigma_a = sigma_b*np.sqrt(z2_bar)
g_fit_hand = b_star/a_star
sigma_g_fit_hand = g_fit_hand*np.sqrt(sigma_a**2/a_star**2 + sigma_b**2/b_star**2)
print('Fit by hand: a = %.2f (%.2f), b = %.2f (%.2f), g = %.2f (%.2f)' % (a_star, sigma_a, b_star, sigma_b, g_fit_hand, sigma_g_fit_hand))

# plot measurements
plt.errorbar(delta_z, ys, sigma_exp, linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.xlabel(r'$\Delta z$ [mm]')
plt.ylabel(r'$I_{out}$ $[W/cm^2]$')

# plot measurements with fit
plt.figure()
plt.errorbar(delta_z, ys, sigma_exp, linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.xlabel(r'$\Delta z$ [mm]')
plt.ylabel(r'$I_{out}$ $[W/cm^2]$')
plt.plot(delta_z, a_star + b_star*delta_z, linewidth = line_width, linestyle = 'dashed')

# plot histogram of measurements
Iins = rng.normal(Iin, sigma_exp/5)
yss = rng.multivariate_normal(ys_clean, np.diag(np.ones(ys_clean.size)*sigma_exp), size = num_hist)
plt.figure()
plt.violinplot(yss)
plt.errorbar(np.arange(1, len(ys) + 1), ys, sigma_exp, linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
plt.xticks(np.arange(1, len(ys) + 1), ['%.1f' % z for z in delta_z])
plt.xlabel(r'$\Delta z$ [mm]')
plt.ylabel(r'$I_{out}$ $[W/cm^2]$')

# MCMC
model = pm.Model()
with model:
    # priors
    I_model = pm.TruncatedNormal('Iin', mu = Iin_0, sigma = sigma_Iin_0, lower = 0)
    g_model = pm.TruncatedNormal('g', mu = g_0, sigma = sigma_g_0, lower = 0)
    
    # deterministic part of model
    mu = Iout(delta_z, I_model, g_model)
    
    # likelihood
    y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma_exp, observed = ys)
    
    # curve fit
    map_estimate = pm.find_MAP(model = model)
    y_fit = Iout(delta_z, map_estimate['Iin'], map_estimate['g'])
    plt.figure()
    print(map_estimate)
    plt.plot(delta_z, y_fit, '--', linewidth = line_width)
    plt.errorbar(delta_z, ys, sigma_exp, linewidth = line_width, capsize = errorbar_capsize, linestyle = '', marker = 'o', capthick = errorbar_capthick)
    plt.plot(delta_z, a_star + b_star*delta_z, linewidth = line_width, linestyle = 'dotted')
    plt.legend(['MAP', 'fit', 'observed'])
    plt.xlabel(r'$\Delta z$ [mm]')
    plt.ylabel(r'$I_{out}$ $[W/cm^2]$')
    plt.title('MAP Estimate fit')
    #text_pos = [np.floor(0.25*decay_df[time_minutes].values[-1]), np.ceil(0.85*np.max(decay_df[analysis_var].values))]
    #rep = 'y = %.2f*exp(-t/%.2f) + %.2f' % (map_estimate_decay['C'], map_estimate_decay['tau'], map_estimate_decay['b'])
    #plt.text(text_pos[0], text_pos[1], rep)
    
    start = map_estimate
    step = pm.NUTS(scaling = start)
    trace_predictive = pm.sample(num_samples, step, start = start, cores = 1)
    #prior = pm.sample_prior_predictive(3000)
    az.plot_trace(trace_predictive)
    az.plot_pair(trace_predictive, marginals = True)
    #az_data = az.from_pymc3(trace_predictive, prior = prior)
    #az.plot_pair(az_data, marginals = True, group = 'prior')

plt.show()
