import os
import csv
import json
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from x15_utils import *
from pathlib import Path

#x15 system globals
n_control             = 6
n_wls                 = 15
n_pixels              = 99
wl_set                = [450 , 520 , 550 , 580 , 610 , 635 , 660 , 700 , 740 , 770 , 810 , 840 , 870 , 900 , 930] 
visible_end_index     = 11 
max_post_adapt_length = 32

colors                = ['b' , 'g' , 'y' , 'r'] 
four_wl_legend        = ['520' , '610' , '700' , '930']
x4_band_indices       = [1,4,7,14]
#x15 system globals

#calib-adapt pixels
pixels_vis_calib_adapt = [4,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

tissue_exclude_list = ['none']
tissue_pick_config =  [['ureter','none'],['peritoneum','none'],['ureter','peritoneum']]
sensor_floor        = 425
watch_baseline      = 1600

if __name__ == '__main__':

    root        = os.getcwd()
    parent      = Path(root)
    parent      = parent.parent.absolute()
    data_path   = os.path.join(parent,'x15_rt')

    children    = glob.glob(os.path.join(data_path,'*x15_rt*'))
    n_children  = len(children)

    print('')
    for i in range(0,len(children)):
        print('Child : ' , children[i])
    print('')

    x15_rt_processed_path    = os.path.join(parent,'x15_rt_processed')
    if not os.path.exists(x15_rt_processed_path):
        os.mkdir(x15_rt_processed_path)

    x15_rt_plots_path = os.path.join(x15_rt_processed_path , 'x15_ext')
    if not os.path.exists(x15_rt_plots_path):
        os.mkdir(x15_rt_plots_path)

    x15_rt_data_path = os.path.join(x15_rt_processed_path, ' accumulated_data')
    if not os.path.exists(x15_rt_data_path):
        os.mkdir(x15_rt_data_path)


    skeletal_ure_file_list  = []
    skeletal_peri_file_list = []
    native_ure_file_list    = []

    for i in range(0,n_children):
        
        child_name              = children[i].split('/')[-1] 
        print('')
        print('Current child : ' , child_name)
        print('')

        targets                 = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets               = len(targets)

        meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
        meta_keys               = meta_keys['keys']

        for j in range(0,n_targets):

            meta = process_x15_rt_file_name(targets[j])
            if meta['baseline'] == str(watch_baseline):

                file_content = process_file_content_x15_rt_two_step(targets[j])
                if not file_content['incomplete_data']:

                    if meta['vessel'] == 'ureter'     and meta['tissue'] == 'none' and meta['position_1'] == '1-99':
                        skeletal_ure_file_list.append(targets[j])
                    if meta['vessel'] == 'peritoneum' and meta['tissue'] == 'none' and meta['position_1'] == '1-99':
                        skeletal_peri_file_list.append(targets[j])
                    if meta['vessel'] == 'ureter'     and meta['tissue'] == 'peritoneum':
                        covered = False
                        position_1 = meta['position_1']
                        position_2 = meta['position_2']

                        p1_start = int(position_1.split('-')[0])
                        p1_end   = int(position_1.split('-')[-1])
                        p2_start = int(position_2.split('-')[0])
                        p2_end   = int(position_2.split('-')[-1])
                        min_start= np.min([p1_start , p2_start])
                        max_end  = np.max([p1_end,p2_end]) 
                        if min_start == 1 and max_end == 99:
                            covered = True
                        if covered:
                            native_ure_file_list.append(targets[j])



    print('Files parsed , categorized. ')
    print('N native ureters       : ' , len(native_ure_file_list))
    print('N skeletal ureters     : ' , len(skeletal_ure_file_list))
    print('N skeletal peritoneums : ' , len(skeletal_peri_file_list))

    skeletal_ures = {}
    skeletal_ures['n'] = len(skeletal_ure_file_list)
    ls = []
    cs = []
    ms = []
    for i in range(0,len(skeletal_ure_file_list)):
        d = {}
        meta = process_x15_rt_file_name(skeletal_ure_file_list[i])
        data = process_file_content_x15_rt_two_step(skeletal_ure_file_list[i])
        l = data['l']
        c = data['x15_c']
        ls.append(l)
        cs.append(c)
        ms.append(meta)

    skeletal_ures['ls'] = ls
    skeletal_ures['cs'] = cs
    skeletal_ures['ms'] = ms

    skeletal_peris = {}
    skeletal_peris['n'] = len(skeletal_peri_file_list)
    ls = []
    cs = []
    ms = []
    for i in range(0,len(skeletal_peri_file_list)):
        d = {}
        meta = process_x15_rt_file_name(skeletal_peri_file_list[i])
        data = process_file_content_x15_rt_two_step(skeletal_peri_file_list[i])
        l = data['l']
        c = data['x15_c']
        ls.append(l)
        cs.append(c)
        ms.append(meta)

    skeletal_peris['ls'] = ls
    skeletal_peris['cs'] = cs
    skeletal_peris['ms'] = ms

    native_ures = {}
    native_ures['n'] = len(native_ure_file_list)
    ls = []
    cs = []
    ms = []
    for i in range(0,len(native_ure_file_list)):
        d = {}
        meta = process_x15_rt_file_name(native_ure_file_list[i])
        data = process_file_content_x15_rt_two_step(native_ure_file_list[i])
        l = data['l']
        c = data['x15_c']
        ls.append(l)
        cs.append(c)
        ms.append(meta)

    native_ures['ls'] = ls
    native_ures['cs'] = cs
    native_ures['ms'] = ms

    sio.savemat('skeletal_ures.mat' , skeletal_ures)
    sio.savemat('skeletal_peris.mat', skeletal_peris)
    sio.savemat('native_ures.mat'   , native_ures)
