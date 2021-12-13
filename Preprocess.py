#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:12:09 2021

@author: mohsenr
"""

from eelbrain import *
import mne
import numpy as np
import os
import scipy.io
import pdb
import csv
import matplotlib.pyplot as plt
import time
from Behavioral import *
from root_folder_list import *
stimchannels = [c for c in range(170,174)]


def makeICA(subject, lowF = 1, hiF = 40, sqdname = 'sns', nica = 0.999, epochflag=False, force_make=False):
    '''
    Makin a file consist of data decomposition usnig ICA. The function save an ica file without a return.
    
    Parameters
    ----------
    subject : str
        Sebject ID to make ICA.
    lowF : float, optional
        lower pass-band for the filter in ICA making. The default is 1.
    hiF : float, optional
        Higher pass-band for the filter in ICA making. The default is 40.
    sqdname : str, optional
        Use 'sns' if the raw data is preprocessed with sns methods. The default is 'sns'.
    nica : float, optional
        number of components required to explain the variance of the data greater than nica. The default is 0.999.
    epochflag : bool, optional
        If True, apply ica on epochs. The default is False.
    force_make : bool, optional
        If True, overwrite an already exist ica, otherwise stop the process. The default is False.

    Returns
    -------
    None.
    '''
    subject_folder = os.path.join(sqd_folder,subject)
    sqd_file = f'{subject_folder}/{subject}_{sqdname}.sqd'
    if not os.path.exists(sqd_file):
        print(f'File not found: {sqd_file}')
    badC_file = f'{subject_folder}/{subject}_badChannels.mat'
    mat = scipy.io.loadmat(badC_file)
    badchannelsAll = mat['badC'][0]
    
    out_folder =  os.path.join(denoise_folder,subject)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if epochflag:
        ica_file = f'{out_folder}/{subject}_{nica}_{lowF}_{hiF}{sqdname}-ica_epochs.fif'
    else:
        ica_file = f'{out_folder}/{subject}_{nica}_{lowF}_{hiF}{sqdname}-ica.fif'
    if force_make or not os.path.exists(ica_file):
        print(f'reading the raw sqd file {sqd_file}')
        raw = mne.io.read_raw_kit(sqd_file, stim_code='channel', stim=stimchannels, preload=True)
    
        print('setting bad channels')
        badchannels = [e + 1 for e in badchannelsAll]
        for i in range(0, len(badchannels)):
            raw.info['bads'].append('MEG %03d' % (badchannels[i]))
        print(raw.info['bads'])
    
        print(f'filtering data {lowF} - {hiF} Hz')
        raw = raw.filter(lowF, hiF)
        
    
        print('running ICA')
        ica = mne.preprocessing.ICA(n_components=nica, method='infomax', max_iter = 'auto')
        print('running ica, this could take a while')
        t = time.time()
        if epochflag:
            ds = load.fiff.events(raw)
            meg = load.fiff.mne_epochs(ds, 0, 6)
            ica.fit(meg)
        else:
            ica.fit(raw)
        elapsed = time.time() - t
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        print(f'time elapsed {round(hrs)}:{round(mins)}:{round(secs)}')
        print(f'saving icafile to {ica_file}')
        ica.save(ica_file)
    else:
        print(f'ICA exists for {subject}_{sqdname}')



def rejectICA(subject: str,ica_sname: str ='0.999_1_40',sqdname: str ='sns', decim: int = 10):
    '''
    Using eelbrain gui function to reject ICA components manually.

    Parameters
    ----------
    subject : str
        Sebject ID to reject ICA components.
    ica_sname : str, optional
        the specific ica file. The default is '0.999_1_40'.
    sqdname : str, optional
        the specific sns denoised data. The default is 'sns'.
    decim : int, optional
        decimate data whith the factor of 'decim'. The default is 10.

    Returns
    -------
    None.

    '''
    subject_folder = f'{sqd_folder}/{subject}'
    sqd_file = f'{subject_folder}/{subject}_{sqdname}.sqd'
    badC_file = f'{subject_folder}/{subject}_badChannels.mat'
    mat = scipy.io.loadmat(badC_file)
    badchannelsAll = mat['badC'][0]

    print(f'reading the raw sqd file {sqd_file}')
    raw = mne.io.read_raw_kit(sqd_file, stim_code='channel', stim=stimchannels, preload=True)

    print('setting bad channels')
    badchannels = [e + 1 for e in badchannelsAll]
    for i in range(0, len(badchannels)):
        raw.info['bads'].append('MEG %03d' % (badchannels[i]))
    print(raw.info['bads'])
    
    datafilter = [float(i) for i in ica_sname.split('_')]
    print(f'filtering data {datafilter[1]}-{datafilter[2]}Hz')
    raw = raw.filter(datafilter[1], datafilter[2])
    denoised_folder =  os.path.join(denoise_folder,subject)
    ica_file = f'{denoised_folder}/{subject}_{ica_sname}{sqdname}-ica.fif'

    ds = load.fiff.events(raw)

    ds['epochs'] = load.fiff.mne_epochs(ds, tmin = -.1, tmax = 6, decim= decim)
    gui.select_components(ica_file, ds)
    
    
    
def Evoked_analysis(subject, lowF = 1, hiF = 40,ica_sname='0.999_1_40',sqdname='sns',subset=False, 
                    baseline=(-.1,0), plotepoch=None, tmin=-.2, tmax=.5, substart=None, subend=None, 
                    decim=1, saveflag = True, output = True):
    '''
    Epoch data, and analyize the evoked responses.

    Parameters
    ----------
    subject : str
        Sebject ID to reject ICA components.
    ica_sname : str, optional
        the specific ica file. The default is '0.999_1_40'.
    sqdname : str, optional
        the specific sns denoised data. The default is 'sns'.
    subset : bool, optional
        If true, chose a subset of trials. The default is False.
    plotepoch : bool | str, optional
        If True plot the evoked TopoButterfly. If it is "cond" plot the evoked 
        TopoButterfly separate for each conditions. The default is None.
    tmin : float, optional
        The time min for the epochs. The default is -.2.
    tmax : float, optional
        The t max for the epochs. The default is .5.
    substart : None | int, optional
        If "subset" is True, trials before "substart" would be discarded. The default is None.
    subend : None | int, optional
        f "subset" is True, trials after "subend" would be discarded. The default is None.
    decim : int, optional
        decimate data whith the factor of 'decim'. The default is 1.
    saveflag : bool, optional
        If True save the epoch files. The default is True.
    output : bool, optional
        If Ture return the epoched data. The default is True.

    Raises
    ------
    EOFError
        If subset is True, MUST specify substart and subend.

    Returns
    -------
    condata : dict
        A dictionary contains epoch data for all conditions.

    '''
    subject_folder = f'{sqd_folder}/{subject}'
    sqd_file = f'{subject_folder}/{subject}_{sqdname}.sqd'
    badC_file = f'{subject_folder}/{subject}_badChannels.mat'
    mat = scipy.io.loadmat(badC_file)
    badchannelsAll = mat['badC'][0]

    print(f'reading the raw sqd file {sqd_file}')
    raw = mne.io.read_raw_kit(sqd_file, stim_code='channel', stim=stimchannels, preload=True)

    print('setting bad channels')
    badchannels = [e + 1 for e in badchannelsAll]
    for i in range(0, len(badchannels)):
        raw.info['bads'].append('MEG %03d' % (badchannels[i]))
    print(raw.info['bads'])
    # TODO: make a filter independent of the ica filter
    datafilter = [float(i) for i in ica_sname.split('_')]
    datafilter[1], datafilter[2] = lowF, hiF
    print(f'filtering data {datafilter[1]}-{datafilter[2]}Hz')
    raw = raw.filter(datafilter[1], datafilter[2])
    denoised_folder =  os.path.join(denoise_folder,subject)
    ica_file = f'{denoised_folder}/{subject}_{ica_sname}{sqdname}-ica.fif'
    if os.path.exists(ica_file):
        ica = mne.preprocessing.read_ica(ica_file)
        print('applying ICA')
        raw = ica.apply(raw)
        no_ica = False
    else:
        print('!!! NO ICA FILE FOUND!!!!!!!!!!!!')
        no_ica = True
    
    ds = load.fiff.events(raw)
    # if subject == 'R2757':
    #     subset = True
    #     substart = 1
    #     subend = 399
    if subset:
        if subend is  None or substart is None:
            raise EOFError("Specify 'substart' and 'subend'!")
        ds = ds.sub(np.arange(substart,subend))  #to discard 7 trials in 7821test 
    csvf = read_log(subject)
    Midx,Fidx,dur,probef,firstidx = read_behave(subject)
    for k in range(len(dur)):
        ds['i_start'][k] += int(dur[k] * raw.info['sfreq']) #Shift trigger to the probe onset
    
    ds['meg'] = load.fiff.epochs(ds, tmin, tmax, decim= decim, baseline=baseline)
    freqs = np.unique(probef)
    if len(freqs) != 4:
        print('WARNING: Should have 4 Probe frequencies!!!!!')
    
    condidx = {}
    for f in range(len(freqs)):
        condidx[f'attM{f}'] = np.intersect1d(np.where(probef == freqs[f]), Midx)
        condidx[f'attF{f}'] = np.intersect1d(np.where(probef == freqs[f]), Fidx)
    
    condata = {}
    for k in condidx.keys():
        condata[k] = ds[condidx[k]]
    # ds[Midx, 'trigger'] = np.zeros(len(Midx))
    # ds[Fidx, 'trigger'] = np.ones(len(Fidx))
    # ds[:,'freq'] = Var(np.zeros(len(ds['trigger'])))
    # for i in range(len(freqs)):
    #     ds[np.where(probef == freqs[i]), 'freq'] = i*np.ones(sum(probef == freqs[i]))
    if baseline!=None:
        bs = 'baseline'
    else:
        bs = ''
    if saveflag:
        print(f'Saving the epoch dtaa into : {subject}_{ica_sname}_{sqdname}_epoch_decim{decim}_{bs}.pkl')
        save.pickle(condata,f'{output_folder}/pickle/{subject}_{ica_sname}_{sqdname}_epoch_decim{decim}_{bs}.pkl')
    if plotepoch:
        labels = {}
        ds[:,'cond'] = Factor([100], repeat=400)
        for i, k in enumerate(condidx.keys()):
            ds[condidx[k], 'cond'] = Factor([i], labels = {i:k})
        if plotepoch == 'cond':
            plot.TopoButterfly('meg', 'cond', ds)
        else:
            plot.TopoButterfly(ds['meg'])
    if output:
        return condata