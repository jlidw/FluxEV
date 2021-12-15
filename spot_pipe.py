#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:08:16 2016

@author: Alban Siffer
@company: Amossys
@license: GNU GPLv3
"""

from scipy.optimize import minimize
from math import log, floor
import numpy as np
import pandas as pd

# colors for plot
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'

"""
================================= MAIN CLASS ==================================
"""


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    data : numpy.array
        stream

    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values

    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4, estimator="MOM"):
        """
        Constructor

	    Parameters
	    ----------
	    q
		    Detection level (risk)
		estimator
		    "MLE": maximum likelihood
		    "MOM": method of moments

	    Returns
	    ----------
    	SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0
        if estimator == "MLE":
            self.estimator = self._grimshaw
        elif estimator == "MOM":
            self.estimator = self._MOM
        else:
            raise TypeError("Unsupported Estimator Type!")

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data):
        """
        Import initial data to SPOT object

        Args:
            init_data: list, numpy.array or pandas.Series.
                       initial batch to calibrate the algorithm

        Returns:

        """

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        else:
            print('The initial data cannot be set')
            return

    def initialize(self, level=0.98, verbose=True):
        """
        Run the calibration (initialization) step

        Args:
            level: float
                  (default 0.98) Probability associated with the initial threshold t
            verbose: bool
                  (default = True) If True, gives details about the batch initialization
        """

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        # if self.Nt == 0:
        #     self.Nt = 1
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self.estimator()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
		    scalar function
        jac : function
            first order derivative of the function
        bounds : tuple
            (min,max) interval for the roots search
        npoints : int
            maximum number of roots to output
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _MOM(self):  # Method of Moments
        Yi = self.peaks
        avg = np.mean(Yi)
        var = np.var(Yi, ddof=1)
        sigma = 0.5 * avg * (1 + avg**2/var)
        gamma = 0.5 * (1 - avg**2/var)
        return gamma, sigma, 0

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
		    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)
            # return self.init_threshold - sigma / gamma

    def run_step(self, data_point):
        """
        Args:
            data_point:

        Returns:

        """

        alarm = 0  # 0: normal point; 1: abnormal point

        # If the observed value exceeds the current threshold (alarm case)
        if data_point > self.extreme_quantile:
            alarm = 1

        # case where the value exceeds the initial threshold but not the alarm ones
        elif data_point > self.init_threshold:
            # we add it in the peaks
            self.peaks = np.append(self.peaks, data_point - self.init_threshold)
            self.Nt += 1
            self.n += 1
            # and we update the thresholds

            g, s, l = self.estimator()
            self.extreme_quantile = self._quantile(g, s)
        else:
            self.n += 1

        th = self.extreme_quantile  # thresholds record

        return th, alarm

