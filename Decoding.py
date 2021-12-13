#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:26:07 2021

@author: mohsenr
"""

from eelbrain import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator, LinearModel, SlidingEstimator, cross_val_multiscore
from Preprocess import *
from root_folder_list import *
from DSS_Analysis import *
from stats import stats


def decoding(subject: str, n_jobs: int=-1, C: float=1, penalty: str='l2', useDSS: bool=False, decim: int = 4,
             npcs: int=25, timegen: bool=False, cv: int=5, plotflag: bool=True, baseline: bool=True,
             ica_sname: str='0.999_1_40',sqdname: str='sns', overwrite:bool=False):
    '''
    Decoding the direction of attention on each trails

    Parameters
    ----------
    subject : str
        Subject ID.
    n_jobs : int, optional
        Number of parallel jobs in cpu. "-1" means all available cores. The default is -1.
    C : float, optional
        Regulation factor. The default is 1.
    penalty : str, optional
        Regulation penalty. "l1" or "l2". The default is 'l2'.
    useDSS : bool, optional
        Use DSS as a dimentionality reduction. The default is False.
    decim : int, optional
       Decimate data after segmenting by factor decim. The default is 4.
    npcs : int, optional
        If useDSS is True, specifies the number of dss components to keep for decoding. The default is 25.
    timegen : bool, optional
        If True generalize the trained decoders to all other times. The default is False.
    cv : int, optional
        Number of cross-validation. The default is 5.
    plotflag : bool, optional
        If true plot the scores of the calssifiers. The default is True.
    ica_sname : str, optional
        the specific ica file. The default is '0.999_1_40'.
    sqdname : str, optional
        the specific sns denoised data. The default is 'sns'.

    Returns
    -------
    scores : NDVar
        Performance scores (2D, or 3D area under the curve matrix) of the trained classifiers.

    '''

    
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        LinearModel(LogisticRegression(penalty=penalty,C=C, solver='liblinear')))
    if useDSS:
        dssname = f'{npcs}dss_{penalty}C{C}'
    else:
        dssname = f'all_{penalty}C{C}'
    if baseline:
        bs = 'baseline'
    else:
        bs = ''
    if timegen:
        decoder = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=n_jobs,
                                       verbose=True)
        decoder_file = (f'{output_folder}/pickle/{subject}_{ica_sname}_{sqdname}'
        +f'_{dssname}_MVPA_decim{decim}_{bs}.pkl')
    else:
        decoder = SlidingEstimator(clf, scoring='roc_auc', n_jobs=n_jobs,
                                       verbose=True)
        decoder_file = (f'{output_folder}/pickle/{subject}_{ica_sname}_{sqdname}'
        +f'_{dssname}_sliding_decim{decim}_{bs}.pkl')
    
    dssFile = f'{output_folder}/pickle/{subject}_{ica_sname}_{sqdname}_{npcs}dss_decim{decim}_{bs}.pkl'
    
    
    if os.path.exists(decoder_file) and not overwrite:
        print('loading SCORES from pickle file')
        scores = load.unpickle(decoder_file)
    else:
        print('Training decoders on DSS file')
        if os.path.exists(dssFile):
            print(f"Loading DSS from Pickle file for decoding: {subject}_{ica_sname}_{sqdname}"
                  +f"_{npcs}dss_decim{decim}_bs.pkl")
            ds = load.unpickle(dssFile)
        else:
            print("!!! No Pickle file Found for DSS!!!!")
            ds = DSS_analysis(subject, npcs = npcs, ica_sname=ica_sname,sqdname=sqdname,
                              decim=decim,baseline=baseline)
        if useDSS:
            X = ds['megdenoised'].x
        else:
            X = ds['meg'].x
        y = ds['attention'].x
        probe_idx = ds['probe'].x
        
        scores=[]
        Midx,Fidx,dur,probef,firstidx = read_behave(subject)
        probes = np.unique(probef)
        for f in range(len(probes)):
            score = cross_val_multiscore(decoder, X=X[probe_idx == f], y=y[probe_idx == f], n_jobs=n_jobs, cv=cv)
            scores.append(score.mean(axis = 0))
        scores = np.asarray(scores)
        if timegen:
            probe_dim = Scalar('probe', np.arange(len(probes)))
            test_time = ds['meg'].time
            train_time = UTS(tmin=test_time.tmin, tstep=test_time.tstep, nsamples=test_time.nsamples)
            train_time.name = 'trainTime'
            scores = NDVar(scores, (probe_dim, test_time, train_time), name = 'decoder')
        else:
            probe_dim = Scalar('probe', np.arange(len(probes)))
            test_time = ds['meg'].time
            scores = NDVar(scores, (probe_dim, test_time), name = 'decoder')
        
        save.pickle(scores, decoder_file)
    Midx,Fidx,dur,probef,firstidx = read_behave(subject)
    probes = np.unique(probef)
    if plotflag:
        for i,f in enumerate(probes):
            figure_folder = f'{root_folder}/Figures/decoder/{dssname}/{f}Hz'
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)
            if timegen:
                fig, ax = plt.subplots(1)
                im = ax.matshow(scores.x[i,:,:], vmin=0.3, vmax=0.7, cmap='RdBu_r', origin='lower', 
                                extent = [scores.time[0], scores.time[-1], scores.time[0], scores.time[-1]])
                ax.axhline(0., color='k')
                ax.axvline(0., color='k')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_title(f'{subject}_{f}Hz')
                plt.colorbar(im, ax=ax)
                plt.savefig(f'{figure_folder}/{subject}_{f}Hz_MVPA_{bs}.pdf', bbox_inches='tight')
                # plt.close('all')
            else:
                t = np.linspace(-.2,.5,scores.x[i,:].shape[0])
                fig, ax = plt.subplots()
                ax.plot(t, scores.x[i,:], label='score')
                ax.axhline(.5, color='k', linestyle='--', label='chance')
                ax.set_xlabel('time')
                ax.set_ylabel('AUC')  # Area Under the Curve
                ax.legend()
                ax.axvline(.0, color='k', linestyle='-')
                ax.set_title(f'{subject}_{f}Hz')
                ax.set_ylim(.1,.9)
                plt.savefig(f'{figure_folder}/{subject}_{f}Hz_Sliding{bs}.pdf', bbox_inches='tight')
    return scores



def plot_decoding(sbjs: list,  n_jobs: int=-1, C: float=1, penalty: str='l2', useDSS: bool=False, decim: int = 4,
             npcs: int=25, timegen: bool=False, cv: int=5, ica_sname: str='0.999_1_40',sqdname: str='sns',
             vmin=0.35, vmax=0.65, plotflag = False, baseline=True, smooth_window = 1e-2):
    scores = []
    if baseline:
        bs = 'baseline'
    else:
        bs = ''
    for subject in sbjs:
        scores.append(decoding(subject, n_jobs=n_jobs, C=C, penalty=penalty, useDSS=useDSS, decim=decim, 
                               npcs=npcs, timegen=timegen, cv=cv, ica_sname=ica_sname, sqdname=sqdname, 
                               plotflag=plotflag, baseline=baseline))
        Midx,Fidx,dur,probef,firstidx = read_behave(subject)
        probes = np.unique(probef)
    
    if useDSS:
        dssname = f'{npcs}dss_{penalty}C{C}'
    else:
        dssname = f'all_{penalty}C{C}'
    figure_folder = f'{root_folder}/Figures/decoder/{dssname}'
    if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)           
    if timegen:
        s = combine(scores).x
        s_mean = combine(scores).mean('case').x
        extent = [combine(scores).time[0], combine(scores).time[-1], combine(scores).time[0],
                  combine(scores).time[-1]]
        for i,f in enumerate(probes):
            p_val = stats(s[:,i,:,:] - .5)
            fig, ax = plt.subplots(1)
            im = ax.matshow(s_mean[i,:,:], vmin=vmin, vmax=vmax, cmap='RdBu_r', origin='lower', extent=extent)
            if p_val.min() < 0.05:
                ax.contour(p_val, levels = [0, 0.05], colors = 'k', linestyles = '--', 
                            linewidths =1,  extent=extent)
            ax.axhline(0., color='k')
            ax.axvline(0., color='k')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xlabel('Testing Time (sample)')
            ax.set_ylabel('Training Time (sample)')
            ax.set_title(f'{int(f)} Hz: p_value = ' + str(round(p_val.min(), 3)))
            plt.colorbar(im, ax=ax)            
            plt.savefig(f'{figure_folder}/Average_{int(f)}Hz_MVPA{bs}.pdf', bbox_inches='tight')
    else:
        from scipy.ndimage import gaussian_filter
        s = combine(scores).smooth('time', window_size=smooth_window, fix_edges=True).x
        s_mean = combine(scores).smooth('time', window_size=smooth_window, fix_edges=True).mean('case').x
        s_sem = combine(scores).smooth('time', window_size=smooth_window, fix_edges=True).std(
            'case').x/np.sqrt(combine(scores).x.shape[0])
        t = 1000*combine(scores).time.times
        for i,f in enumerate(probes):
            p_val = stats(s[:,i,1:-1]-.5)
            p_05 = np.ma.masked_where(p_val>0.05, s_mean[i,1:-1])
            fig, ax = plt.subplots()
            ax.plot(t[1:-1], s_mean[i,1:-1], label='scores', lw = .5)
            ax.plot(t[1:-1], p_05, c = [0., 0.4, .6], lw = 2)
            ax.fill_between(t[1:-1], s_mean[i,1:-1]-s_sem[i,1:-1], s_mean[i,1:-1]+s_sem[i,1:-1], alpha = 0.3) 
            ax.axhline(.5, color='k', linestyle='--', label='chance')
            ax.set_xlabel('Time(ms)')
            ax.set_ylabel('AUC')  # Area Under the Curve
            ax.axvline(.0, color='k', linestyle='-')
            ax.set_title(f'{int(f)} Hz: p_value = ' + str(round(p_val.min(), 3)))
            ax.set_xlim(-200,500)
            ax.set_ylim(vmin,vmax)
            plt.savefig(f'{figure_folder}/Average_{int(f)}Hz_Sliding{bs}.pdf', bbox_inches='tight')