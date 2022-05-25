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
calib_band_indices     = [4,14]
pixels_vis_calib_adapt = [4,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

tissue_exclude_list = ['none']
sensor_floor        = 425

watch_baseline      = 1600
vertical_lines      = 1

def plot_refs_for_paul(c , title , save = True , show = True):

    c = c - 400
    c = c/3696

    global calib_band_indices , watch_baseline , pixels_vis_calib_adapt , pixels_nir_calib_adapt , n_pixels
    plt.figure()
    plt.xlabel('pixel count')
    plt.ylabel('Normalized cmos intensity')
    plt.ylim([0,1])
    plt.title('Reference bands at adapt end.')
    #plt.title(title)
    plt.plot(c[calib_band_indices[0],:],'b-')
    plt.plot(c[calib_band_indices[1],:],'r-')
    plt.plot(np.ones([n_pixels]) * (watch_baseline-400)/3696,'g-')
    for j in range(0,len(pixels_vis_calib_adapt)):
        plt.axvline(pixels_vis_calib_adapt[j],color='b')
    for j in range(0,len(pixels_nir_calib_adapt)):
        plt.axvline(pixels_nir_calib_adapt[j],color='r')
    if save:
        plt.savefig('x.pdf')
    if show:
        plt.show()

    return

def plot_spatial_for_paul(c , title , save = True , show = True):
    
    global watch_baseline , pixels_vis_calib_adapt , pixels_nir_calib_adapt , n_pixels

    c = c - 400
    c = c/3696

    plt.figure()
    plt.xlabel('pixel count')
    plt.ylabel('Normalized cmos intensity')
    plt.ylim([0,1])
    #plt.title(title)
    plt.title('Average energy')
    plt.plot(np.mean(c,axis = 0),'y-')
    plt.plot(np.ones([n_pixels]) * (watch_baseline-400)/3696,'g-')
    for j in range(0,len(pixels_vis_calib_adapt)):
        plt.axvline(pixels_vis_calib_adapt[j],color='b')
    for j in range(0,len(pixels_nir_calib_adapt)):
        plt.axvline(pixels_nir_calib_adapt[j],color='r')
    if save:
        plt.savefig('x.pdf')
    plt.show()

    return

def plot_spectral_for_paul(c,title,save = True , show = True):

    global watch_baseline , pixels_vis_calib_adapt , pixels_nir_calib_adapt , n_wls , wl_set
    c = c - 400
    c = c/3696
    plt.figure()
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Normalized mean cmos intensity')
    plt.ylim([0,1])
    #plt.title(title)
    plt.title('Mean spectral response')
    plt.plot(wl_set , np.mean(c,axis = 1),'y-')
    plt.plot(wl_set , np.ones([n_wls]) * (watch_baseline-400)/3696,'g-')
    if save:
        plt.savefig('x.pdf')
    plt.show()

    return

if __name__ == '__main__':

    root        = os.getcwd()
    parent      = Path(root)
    parent      = parent.parent.absolute()
    data_path   = os.path.join(parent,'x15_rt')

    child       = os.path.join(data_path,'2022315_x15_rt')
    targets     = glob.glob(os.path.join(child,'*.csv'))

    relevant_files = []
    n_targets = len(targets)
    for i in range(0,n_targets):
        meta = process_x15_rt_file_name(targets[i])
        if meta['baseline'] == str(watch_baseline):
            relevant_files.append(targets[i])
    n_relevant_files = len(relevant_files)
    
    for i in range(0,n_relevant_files):

        meta = process_x15_rt_file_name(relevant_files[i])
        data = process_file_content_x15_rt_two_step(relevant_files[i])

        leds = data['l']
        cmos = data['x15_c']

        title_parent = meta['comment'].split('for_pc_')[-1]

        print('Sample : ' , title_parent)

        plot_refs_for_paul(    cmos ,title_parent,save = False , show = True)

        plot_spatial_for_paul( cmos ,title_parent,save = False , show = True)

        plot_spectral_for_paul(cmos,title_parent,save = False , show = True)

        #for the shape making ,
        #is there a simple feedback loop / program I can write ?
        #to show the flexibility of the adapt ?  