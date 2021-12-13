#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:42:07 2021

@author: mohsenr
"""

root_folder = '/auto/data/EEG/After_COVID/Streaming/Fricative_Exp'
sqd_folder = f'{root_folder}/data/MEGraw'
denoise_folder = f'{root_folder}/data/MEGdenoise/'
code_folder = f'{root_folder}/Analysis/'
output_folder = f'{root_folder}/Results'
log_folder = f'{root_folder}/data/Log'
Source_folder = f'{root_folder}/data/SourceSpace'



Emptyroom = {'R2605': 'ERT_08.12.21_10.48',
             'R2702': 'ERT_08.10.21_16.15',
             'R2703': 'ERT_08.11.21_16.32',
             'R2704': 'ERT_08.11.21_16.32',
             'R2707': 'ERT_08.23.21_14.10',
             'R2708': 'ERT_08.23.21_14.10',
             'R2709': 'ERT_08.23.21_14.10',
             'R2711': 'ERT_08.23.21_14.10',
             'R2716': 'ERT_08.26.21_14.05',
             'R2722': 'ERT_09.03.21_12.15',
             'R2723': 'ERT_09.03.21_12.15',
             'R2724': 'ERT_09.08.21_15.50',
             'R2725': 'ERT_09.08.21_15.50',
             'R2726': 'ERT_09.13.21_19.00',
             'R2727': 'ERT_09.13.21_19.00',
             'R2730': 'ERT_09.13.21_19.00',
             'R2731': 'ERT_09.20.21_12.25',
             'R2732': 'ERT_09.20.21_12.25',
             'R2733': 'ERT_09.20.21_12.25',
             'R2734': 'ERT_09.20.21_12.25',
             'R2738': 'ERT_09.23.21_18.00',
             'R2741': 'ERT_09.27.21_17.00',
             'R2742': 'ERT_09.27.21_17.00',
             'R2742': 'ERT_09.27.21_17.00',
             'R2750': 'ERT_10.04.21_12.25',
             'R2751': 'ERT_10.04.21_12.25',
             'R2755': 'ERT_10.08.21_15.45',
             'R2757': 'ERT_10.08.21_15.45'
             }


Silence = {'R2605': .01,
            'R2702': .01,
            'R2703': .01,
            'R2704': .01,
            'R2707': .01,
            'R2708': .01,
            'R2709': .01,
            'R2711': .01,
            'R2716': .16,
            'R2722': .16,
            'R2723': .16,
            'R2724': .16,
            'R2725': .16,
            'R2726': .16,
            'R2727': .16,
            'R2730': .16,
            'R2731': .16,
            'R2732': .16,
            'R2733': .16,
            'R2734': .16,
            'R2738': .16,
            'R2741': .16,
            'R2742': .16,
            'R2743': .16,
            'R2750': .16,
            'R2751': .16,
            'R2755': .16,
            'R2757': .16,
            }

