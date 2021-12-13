#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:16:11 2021

@author: mohsenr
"""

from eelbrain import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from Preprocess import *
from root_folder_list import *
stimchannels = [c for c in range(170,174)]



def dss_nd(data: NDVar,thr: float = 1e-11, topoplot: int = None, show_scores: bool = True):
    '''
    Perform Denoised Source Separation (DSS) method on an NDVar.

    Parameters
    ----------
    data : NDVar
        Data to be denoised .
    thr : float, optional
        Ignore PCs smaller than "thr". The default is 1e-11.. The default is 1e-11.
    topoplot : None | int, optional
        If a number, plot topomaps for the specified number of components. The default is None.
    show_scores : bool, optional
        If True plot the repeatability scores for all components. The default is True.

    Returns
    -------
    todss : NDVar
        A matrix to transform sensor space data to DSS.
    fromdss : NDVar
        A matrix to project back DSSed data to sensor space.

    '''
    bias = data.mean(axis = 'case')
    todss, scores = _dss_np(data.x,bias.x,thr = thr)

    dss_dim = Scalar('dss', np.arange(todss.shape[0]))
    data_dim = data.sensor
    todss = NDVar(todss, (dss_dim, data_dim), {}, 'to dss')
    fromdss = NDVar(np.linalg.pinv(todss), (data_dim, dss_dim), {}, 'from dss')
    if show_scores:
        fig, ax = plt.subplots()
        ax.plot(scores, marker = '.', markersize = 15)
        ax.set_ylabel('scores')
        ax.set_xlabel('DSS Components')
        ax.set_title('Repeatability Score')
    if topoplot:
        plot.Topomap(fromdss[:, :topoplot], '.dss', ncol=5, h=6)
    return todss, fromdss

def _dss_np(data,bias,thr):
    '''
    This function applies to numpy array data to dss denoise

    Parameters
    ----------
    data : np array: [ntrials, nsensors, ntimes]
           Data to denoise.
           
    bias : np array
           Average over the trials (Bias function)
           
    thr : float, optional
        ignore PCs smaller than "thr". The default is 1e-11.

    Returns
    -------
    to_dss : np array
        denoising matrix.
    from_dss : np array
        Psudo-invers of denoising matrix.

    '''
    
    if data.ndim != 3:
            raise ValueError('Data must have shape '
                             '[ntrials, nsensors, ntimes].')
    
    demean = lambda x: x - x.mean(axis = 1, keepdims=True)
    normalize = lambda x: x/x.max()
    
    data = normalize(demean(data)) # remove the mean and normalize the data
    bias = normalize(demean(bias))
    
    data_cov = np.sum([np.dot(trial, trial.T) for trial in data], axis=0)
    bias_cov = np.cov(bias)
    
    data_eigval, data_eigvec = pca_np(data_cov, thr)
    W = np.sqrt(1 / data_eigval)  # apply PCA and whitening to the biased covariance
    c2 = (W * data_eigvec).T.dot(bias_cov).dot(data_eigvec) * W
    # matrix to convert PCA-whitened data to DSS
    bias_eigval, bias_eigvec = pca_np(c2, thr)
    # DSS matrix (raw data to normalized DSS)
    todss = (W[np.newaxis, :] * data_eigvec).dot(bias_eigvec)
    # normalize DSS dimensions
    N = np.sqrt(1 / np.diag(todss.T.dot(data_cov).dot(todss))) 
    todss = (N * todss).T
    # power per DSS component
    pwr0=np.sqrt(sum((data_cov.T.dot(todss.T))**2)) #unbiased
    pwr1=np.sqrt(sum((bias_cov.T.dot(todss.T))**2)) #biased
    
    print(f'Number of DSS components: {todss.shape[0]}')
    #return toDSS and Scores for each component
    return todss, pwr1/pwr0

def pca_np(cov, thr):
    """
    Perform PCA decomposition
    Parameters
    ----------
    cov : numpy array [nsensors, nsensors]
        Covariance matrix
    thr : float
        ignore PCs smaller than thr (default: 1e-11)
    Returns
    -------
    eigval : np array
        eigenvalues 1D array.
    eigvec : np array
        eigenvectors 2D array.
    """

    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.abs(eigval)
    sort_ix = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, sort_ix]
    eigval = eigval[sort_ix]
    if thr is not None:
        suprathresh = np.where(eigval / eigval.max() > thr)[0]
        eigval = eigval[suprathresh]
        eigvec = eigvec[:, suprathresh]
    return eigval, eigvec





def DSS_analysis(subject: str, npcs:int=25, dsstopo:int=None, show_scores:bool=False, decim:int=4, baseline=True,
                 saveflag: bool = True,ica_sname: str ='0.999_1_40',sqdname: str ='sns', overwrite:bool=False,
                 tmin:float=-.2, tmax:float=.5):
    '''
    Perform the DSS analysis on a subject data.

    Parameters
    ----------
    subject : str
        Sebject ID to reject ICA components.
    npcs : int, optional
        Number of dss components to keep. The default is 25.
    dsstopo : int, optional
        If a number, plot topomaps for the specified number of components. The default is None.
    show_scores : bool, optional
        If True plot the repeatability scores for all components. The default is False.
    decim : int, optional
       Decimate data after segmenting by factor decim. The default is 4.
    saveflag : bool, optional
        If True svae a pickle file from a dataset. The default is True.
    ica_sname : str, optional
        the specific ica file. The default is '0.999_1_40'.
    sqdname : str, optional
        the specific sns denoised data. The default is 'sns'.
    tmin : float, optional
        The time min for the epochs. The default is -.2.
    tmax : float, optional
        The t max for the epochs. The default is .5.

    Returns
    -------
    Data set
        A Data set, contains the all the meg data, dss data, and denoised data.

    '''
    if baseline:
        bs = 'baseline'
        bb = (-.1, 0)
    else:
        bs = ''
        bb = None
    pkl_file = f'{output_folder}/pickle/{subject}_{ica_sname}_{sqdname}_epoch_decim1_{bs}.pkl'
    
    if os.path.exists(pkl_file) and not overwrite:
        print("Loading Pickle file for DSS analysis")
        epoch = load.unpickle(pkl_file)
    elif os.path.exists(pkl_file) and overwrite:
        print('Overwrite the saved pickle file')
        epoch = Evoked_analysis(subject,ica_sname=ica_sname,sqdname=sqdname, tmin=tmin,
                                tmax=tmax, baseline=bb)
    else:
        print("!!! No Pickle file Found from epoch!!!!")
        epoch = Evoked_analysis(subject,ica_sname=ica_sname,sqdname=sqdname, tmin=tmin,
                                tmax=tmax, baseline=bb)
    tstart, tstop, nsamples = -tmin*1000, tmax*1000, int((tmax-tmin)*1000 +1)
    alldata = combine([epoch[k]['meg'] for k in epoch.keys()])
    todss, fromdss = dss_nd(alldata, topoplot = dsstopo, show_scores=show_scores)
    del alldata
    for k in epoch.keys():
        z = todss[:npcs,:].dot(concatenate(epoch[k]['meg'], 'time'))
        xx = fromdss[:,:npcs].dot(z)
        times = Scalar('case', np.arange(tstart,len(z.time),nsamples)/1000)
        epoch[k]['dss'] = segment(z, times, tmin, .001*(tstop+1), decim = decim)
        epoch[k]['megdenoised'] = segment(xx, times, tmin, .001*(tstop+1), decim = decim)
        epoch[k]['meg'] = segment(concatenate(epoch[k]['meg'], 'time'), times,
                                  tmin, .001*(tstop+1), decim=decim)
        n_case = epoch[k].n_cases
        if k[-2] == 'M':
            epoch[k]['attention'] = Factor([0], labels={0:'Male'}, repeat=n_case)
        else:
            epoch[k]['attention'] = Factor([1], labels={1:'Female'}, repeat=n_case)
        if k[-1] == '0':
            epoch[k]['probe'] = Factor([0], repeat = n_case)
        elif k[-1] == '1':
            epoch[k]['probe'] = Factor([1], repeat = n_case)
        elif k[-1] == '2':
            epoch[k]['probe'] = Factor([2], repeat = n_case)
        else:
            epoch[k]['probe'] = Factor([3], repeat = n_case)
    
    if saveflag:
        print(f'Saving the dss data into : {subject}_{ica_sname}_{sqdname}_{npcs}dss_decim{decim}_{bs}.pkl')
        save.pickle(combine([epoch[k] for k in epoch.keys()]),
                   f'{output_folder}/pickle/{subject}_{ica_sname}_{sqdname}_{npcs}dss_decim{decim}_{bs}.pkl')
    
    return combine([epoch[k] for k in epoch.keys()])