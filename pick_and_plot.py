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

reference_keywords  = ['dailytest' , 'daily_test']
tissue_exclude_list = ['none']
tissue_pick_config  = ['nerve','none']
#add nerve to list of vessels and tissues

sensor_floor        = 425

watch_baseline      = 2000
vertical_lines      = 0

#4  point plots
#15 point plots
#Slopes
#Air-reference Normalized

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

    x15_rt_processed_path = os.path.join(x15_rt_processed_path , 'tissue_configs')
    if not os.path.exists(x15_rt_processed_path):
        os.mkdir(x15_rt_processed_path)

    x15_plots_path = os.path.join(x15_rt_processed_path , tissue_pick_config[0])
    if not os.path.exists(x15_plots_path):
        os.mkdir(x15_plots_path)

    spectral_plots_path = os.path.join(x15_plots_path , 'spectral')
    spatial_plots_path  = os.path.join(x15_plots_path , 'spatial')
    if not os.path.exists(spectral_plots_path):
        os.mkdir(spectral_plots_path)
    if not os.path.exists(spatial_plots_path):
        os.mkdir(spatial_plots_path)

    for i in range(0,n_children):
        
        child_name              = children[i].split('/')[-1] 

        print('')
        print('Current child : ' , child_name)
        current_plots_folder_spectral   = os.path.join(spectral_plots_path,child_name)
        current_plots_folder_spatial    = os.path.join(spatial_plots_path,child_name)
        if not os.path.exists(current_plots_folder_spectral):
            os.mkdir(current_plots_folder_spectral)
        if not os.path.exists(current_plots_folder_spatial):
            os.mkdir(current_plots_folder_spatial)

        targets                 = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets               = len(targets)

        meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
        meta_keys               = meta_keys['keys']

        #For each folder , check if the category exists , make raw and all norm plots.
        #Have normalization functions / lambdas

        process_list            = []

        for j in range(0,n_targets):

            current_meta      = process_x15_rt_file_name(targets[j])
            current_primary   = current_meta['vessel']  #a string.
            current_baseline  = current_meta['baseline']#a string.

            if current_primary == tissue_pick_config[0] and current_baseline == str(watch_baseline):
                process_list.append(targets[j])

        print('Files to process.')
        n_process_list = len(process_list)
        for j in range(0,n_process_list):
            print(process_list[j])
        print('')


        for j in range(0,n_process_list):

            current      = process_list[j]
            current_meta = process_x15_rt_file_name(process_list[j])
            current_plot_file_name = current_meta['vessel'] + '_' + \
                                     current_meta['tissue'] + '_' + \
                                     current_meta['ja_manual'] + '_' + \
                                     current_meta['baseline'] + '.pdf'

            roi       = current_meta['position_1']
            roi_start = int(roi.split('-')[0])
            roi_end   = int(roi.split('-')[-1])
            
            if roi_start < 0:
                roi_start = 0
            if roi_end > n_pixels:
                roi_end   = n_pixels
            
            roi_centre = int(0.5*(roi_start + roi_end))

            left_roi  = int(0.5 * (roi_start))
            if left_roi < 8:
                left_roi = 12
            
            right_roi = int(0.5 * (roi_end + n_pixels))
            if right_roi > n_pixels:
                right_roi = 80

            #plot spatial 
            #plot spectral , at 3 points , in and out of the RoI

            current_content = process_file_content_x15_rt_two_step(process_list[j])
            current_cmos    = current_content['x15_c']
            current_leds    = current_content['l']
            if not current_content['incomplete_data']:
                legend = ['left of roi' , 'roi mid point' , 'right of roi' , 'baseline']
            
                #plot raw signal levels.
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title( 'Tx intensities (Spectral) - Nerve , Not nerve')
                plt.ylim(  [256,4096])
                
                plt.plot(wl_set,current_cmos[:,left_roi])
                plt.plot(wl_set,current_cmos[:,roi_centre])
                plt.plot(wl_set,current_cmos[:,right_roi])
                plt.plot(wl_set,np.ones([n_wls]) * watch_baseline)

                for j in range(0,len(wl_set)):
                    plt.axvline(wl_set[j])
                plt.legend(legend)
                
                data_save_at = os.path.join(current_plots_folder_spectral,current_plot_file_name)
                plt.savefig(data_save_at)
                #plot raw signal levels.
                plt.show()


                plt.figure()
                legend = ['610' , '930' , 'baseline' , 'roi_start' , 'roi_end']
                plt.xlabel('pixel count')
                plt.ylabel('cmos intensity')
                plt.title( 'Tx intensities (Spatial) - Nerve , Not nerve')
                plt.ylim(  [256,4096])

                plt.plot(current_cmos[4,:])
                plt.plot(current_cmos[14,:])
                plt.plot(np.ones([n_pixels]) * watch_baseline)
                plt.axvline(roi_start)
                plt.axvline(roi_end)
                
                plt.legend(legend)

                data_save_at = os.path.join(current_plots_folder_spatial,current_plot_file_name)
                plt.savefig(data_save_at)

                plt.show()
