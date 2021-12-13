#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:13:41 2021

@author: mohsenr
"""
import numpy as np
import os
import csv
from root_folder_list import *


def read_log(subject: str):
    '''
    Read the log (csv format) file from subjects behavioral responses
    Parameters
    ----------
    subject : str
        Subject ID.

    Returns
    -------
    csv_list
        A list contains all behavioral and log information.

    '''
    csv_in = os.path.join(log_folder,subject) + f'/{subject}.csv'
    with open(csv_in) as csvfile:
        csvf = csv.reader(csvfile, delimiter='\t')
        return [rwo for rwo in csvf]
def read_behave(subject: str):
    '''
    Generate the trials specific information .

    Parameters
    ----------
    subject : str
        Subject ID.

    Returns
    -------
    Midx : list
        Indeces for attend male trials.
    Fidx : list
        Indeces for attend male trials.
    Duration : np array
        Duration of each trial.
    probef : np array
        Probe frequency at each trial.
    firstidx : list
        Specifying the starting voice in each trial.

    '''
    silence = Silence[subject]
    csv_list = read_log(subject)
    if subject == 'R2757':
        del(csv_list[7])
    Midx, Fidx, dur, probef, firstidx = [], [], [], [], []
    iatt = csv_list[5].index('Attend')
    if csv_list[5].count('num_syllables'):
        idur = csv_list[5].index('num_syllables')
    else:
        idur = csv_list[5].index('num_syllables_male')
    iprobef = csv_list[5].index('probe_freq')
    ifirst = csv_list[5].index('first_speaker')
    for itrial in range(len(csv_list[7:])):
        if csv_list[itrial+7][iatt] == 'M':
            Midx.append(itrial)
            dur.append(int(csv_list[itrial+7][idur])*2*.16 + silence)
        else:
            Fidx.append(itrial)
            dur.append(int(csv_list[itrial+7][idur])*2*.16 + .16 + silence)
        probef.append(float(csv_list[itrial+7][iprobef]))
        firstidx.append(csv_list[itrial+7][ifirst])
    return Midx,Fidx, np.asarray(dur), np.asarray(probef), firstidx
   
def behave_analysis(subject: str):
    '''
    Analyze the behavioral performance of subjects

    Parameters
    ----------
    subject : str
        Subject ID.

    Returns
    -------
    d-prime.

    '''
    from scipy.stats import norm
    Z = norm.ppf
    
    csv_list = read_log(subject)
    iresult = csv_list[5].index('Result')
    iresp = csv_list[5].index('ButtonCode')
    Hit, Miss, CR, FA = 0, 0, 0, 0
    for itrial in range(len(csv_list[7:])):
        if csv_list[itrial+7][iresult]=='correct' and csv_list[itrial+7][iresp]=='5':
            Hit+=1
        elif csv_list[itrial+7][iresult]=='correct' and csv_list[itrial+7][iresp]=='6':
            CR += 1
        elif csv_list[itrial+7][iresult]=='wrong' and csv_list[itrial+7][iresp]=='6':
            Miss += 1
        else:
            FA += 1
    HitRate = Hit/(Hit + Miss)
    FArate = FA/(FA + CR)
    
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = .5/(Hit + Miss)
    half_fa = .5/(FA + CR)
 
    # Calculate hit_rate and avoid d' infinity
    if HitRate == 1: 
        HitRate = 1 - half_hit
    if HitRate == 0: 
        HitRate = half_hit

    if FArate == 1: 
        FArate = 1 - half_fa
    if FArate == 0: 
        FArate = half_fa
        
    return Z(HitRate) - Z(FArate)