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
n_control              = 6
n_wls                  = 15
n_pixels               = 99
wl_set                 = [450 , 520 , 550 , 580 , 610 , 635 , 660 , 700 , 740 , 770 , 810 , 840 , 870 , 900 , 930] 
visible_end_index      = 11 
max_post_adapt_length  = 32

colors                 = ['b' , 'g' , 'y' , 'r'] 
four_wl_legend         = ['520' , '610' , '700' , '930']
x4_band_indices        = [1,4,7,14]
#x15 system globals

#calib-adapt pixels
pixels_vis_calib_adapt = [4 ,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

tissue_exclude_list    = [['none','none']]
sensor_floor           = 425
watch_baseline         = 1800
vertical_lines         = 0

reference_keywords  = ['dailytest' , 'daily_test']

if __name__ == '__main__':

    #Look at the distribution of 
    #LED states and the distribution of signal levels

    #Adapt with only LED 3 and LED 4
    #Adapt at 1600,1800,2000

    #A truly dynamic process
    #Dropout like mechanisms 

    #Adapt individually ? (pixels)

    #Run the adapt once - > Have a free run around that point ? 
    #For example , using above , collect for 2 minutes ? 

    #In the final illumination scheme , average of regions always ? 

    #Calculate ideal LED placements and beamwidths

    #Role changing - untied , but dependent LEDs/controls

    #"Among yourselves , find a configuration that would establish a 
    #certain signal shape , under the following constraints."

    #Initial update program - Calibration curve following. 
    #Dynamically update the update program. Has to be interruptible
    #and 'controllable'. What is a program ? Has to be a differential 
    #program.

    #I already have in x15_utils , a function to plot the avg of an RoI
    #of all 15 bands.
    #Write a function to plot the distribution of LEDs : WL and NIR
    #along the adapt sequence. Pass the LEDs as a list of arrays.

    root        = os.getcwd()
    parent      = Path(root)
    parent      = parent.parent.absolute()
    data_path   = os.path.join(parent,'x15_rt')
    data_path   = os.path.join(data_path , '2022504_x15_rt')

    targets     = glob.glob(os.path.join(data_path,'*.csv'))
    n_targets   = len(targets)

    print('Targets found : ' , n_targets)

    meta_keys   = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
    meta_keys   = meta_keys['keys']

    unique_configurations   = []
    led_distributions       = []
    air_references          = [] 
    categorized_lists       = {}

    for j in range(0,n_targets):

        current_meta = process_x15_rt_file_name(targets[j])
        current_primary = current_meta['vessel']
        if current_primary not in unique_configurations:
            unique_configurations.append(current_primary)


    for category in unique_configurations:

        s = []
        for j in range(0,n_targets):
            current_meta = process_x15_rt_file_name(targets[j])
            if current_meta['baseline'] == str(watch_baseline) and current_meta['vessel'] == category:
                s.append(targets[j])
        categorized_lists[category] = s
    categorized_lists['n_categories'] = len(unique_configurations)

    for key in categorized_lists:

        print('Key : ' , key)
        if key not in ['n_categories']:
            for j in range(0,len(categorized_lists[key])):
                print(categorized_lists[key][j])
        print('')

    for key in categorized_lists:

        if key not in ['n_categories']:

            comments   = []
            cmos       = []
            leds_final = []
            leds_all   = []

            n_samples = len(categorized_lists[key])
            for j in range(0,n_samples):
                current_meta = process_x15_rt_file_name(categorized_lists[key][j])
                current_data = process_file_content_x15_rt_two_step(categorized_lists[key][j])

                comment = current_meta['comment']
                #find air ref data here.

                if not current_data['incomplete_data']:
                    comments.append(comment)
                    cmos.append(current_data['x15_c'])
                    leds_final.append(current_data['l'])
                    leds_all.append(np.array(current_data['x15_ls']))

            #Spatial , Visible band.
            plt.figure()
            plt.title('Visible reference bands : ' + key)
            plt.ylim([256,4096])
            plt.ylabel('Raw cmos intensity')
            plt.xlabel('Pixel count')
            for j in range(0,len(cmos)):
                plt.plot(cmos[j][4,:])
            plt.legend(comments)
            plt.savefig('vis_ref_def.pdf')
            plt.show()
            #Spatial , visible band.

            #Spatial , NIR band.
            plt.figure()
            plt.title('NIR reference bands : ' + key)
            plt.ylim([256,4096])
            plt.ylabel('Raw cmos intensity')
            plt.xlabel('Pixel count')
            for j in range(0,len(cmos)):
                plt.plot(cmos[j][14,:])
            plt.legend(comments)
            plt.savefig('nir_ref_def.pdf')
            plt.show()
            #Spatial , NIR band.

            #Spectral
            plt.figure()
            plt.title('Spectral intensities : ' + key)
            plt.ylim([256,4096])
            plt.ylabel('Raw cmos intensity')
            plt.xlabel('Wavelength')
            for j in range(0,len(cmos)):
                plt.plot(wl_set , np.mean(cmos[j][:,0:48],axis = 1))
            plt.legend(comments)
            plt.savefig('spec_def.pdf')
            plt.show()
            #Spectral

            #LED distributions
             
            #LED distributions
                    

