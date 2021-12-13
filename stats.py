#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17  16:41:34 2021

@author: mohsenr
"""
import numpy as np
from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p

def _stat_fun(x, sigma=0, method='relative'):
    """Aux. function of stats"""
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def stats(X, n_jobs=-1):
    """Cluster statistics to control for multiple comparisons.
    Parameters
    ----------
    X : array, shape (n_samples, n_space, n_times)
        The data, chance is assumed to be 0.
    n_jobs : int
        The number of parallel processors.
    """
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=_stat_fun, n_permutations=2**12,
        n_jobs=n_jobs)
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T


def run_permutation_test(pooled,sizeZ,sizeY,delta):
    np.random.shuffle(pooled)
    starZ = pooled[:sizeZ]
    starY = pooled[-sizeY:]
    return starZ.mean() - starY.mean()