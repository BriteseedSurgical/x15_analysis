import os
import csv

import glob
import json

import numpy as np
import pandas as pd
import scipy.io as sio

from pathlib import Path
import matplotlib.pyplot as plt

from x15_utils import *

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

#lambdas
vis_read = lambda x: (int(x[1] + x[3] + x[5]) == int(48))
nir_read = lambda x: (int(x[0] + x[2] + x[4]) == int(48))
#lambdas

#calib-adapt pixels
pixels_vis_calib_adapt = [4,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

start_intensity = 200
stop_intensity  = 3000

seek_leds_set_2 = [500,700,1000,1500,2000,2500]

vis_band_index    = 4
nir_band_index    = 14
seek_pixels_set_1 = [25,50,90]

if __name__ == '__main__' : 

    print('')
    root            = os.getcwd() 
    parent          = Path(root)
    parent          = parent.parent.absolute()
    
    calib_data_path = os.path.join(parent , 'x15_calib')
    calib_data_path = os.path.join(calib_data_path,'2022208_x15_calib')

    x               = 'light_off_mask_no_blocker_no'
    y               = 'light_off_mask_yes_blocker_no'

    target_1        = os.path.join(calib_data_path,x)
    target_2        = os.path.join(calib_data_path,y)

    targets_1 = glob.glob(os.path.join(target_1,'*.csv'))
    targets_2 = glob.glob(os.path.join(target_2,'*.csv'))

    calib_light_off  = read_calib_data(target_1,vis_band_index,seek_pixels_set_1,nir_band_index,seek_pixels_set_1)
    calib_light_on   = read_calib_data(target_2,vis_band_index,seek_pixels_set_1,nir_band_index,seek_pixels_set_1)


    ###
    d_off_vis = calib_light_off['135']
    l_off_vis = d_off_vis['l']

    start_index = 0
    stop_index  = 0
    for j in range(0,np.shape(l_off_vis)[0]):
        if l_off_vis[j] >= start_intensity:
            start_index = j
            break
    for j in range(0,np.shape(l_off_vis)[0]):
        if l_off_vis[j] >= stop_intensity:
            stop_index = j-1
            break

    l_off_vis = l_off_vis[start_index:stop_index]
    c_off_vis = [d_off_vis['p_' + str(seek_pixels_set_1[k]) + '_b_' + str(wl_set[4])][start_index:stop_index] for k in range(0,3)]

    d_off_nir = calib_light_off['246']
    l_off_nir = d_off_nir['l']

    start_index = 0
    stop_index  = 0
    for j in range(0,np.shape(l_off_nir)[0]):
        if l_off_nir[j] >= start_intensity:
            start_index = j
            break
    for j in range(128,np.shape(l_off_nir)[0]):
        if l_off_nir[j] >= stop_intensity:
            stop_index = j-1
            break
    
    l_off_nir = l_off_nir[start_index:stop_index]
    print('Case 1 : ' , start_index , stop_index)
    c_off_nir = [d_off_nir['p_' + str(seek_pixels_set_1[k]) + '_b_' + str(wl_set[14])][start_index:stop_index] for k in range(0,3)]
    ###


    ###
    d_on_vis = calib_light_on['135']
    l_on_vis = d_on_vis['l']

    start_index = 0
    stop_index  = 0
    for j in range(0,np.shape(l_on_vis)[0]):
        if l_on_vis[j] >= start_intensity:
            start_index = j
            break
    for j in range(0,np.shape(l_on_vis)[0]):
        if l_on_vis[j] >= stop_intensity:
            stop_index = j-1
            break

    l_on_vis = l_on_vis[start_index:stop_index]
    c_on_vis = [d_on_vis['p_' + str(seek_pixels_set_1[k]) + '_b_' + str(wl_set[4])][start_index:stop_index] for k in range(0,3)]


    d_on_nir = calib_light_on['246']
    l_on_nir = d_on_nir['l']

    start_index = 0
    stop_index  = 0
    for j in range(0,np.shape(l_on_nir)[0]):
        if l_on_nir[j] >= start_intensity:
            start_index = j
            break
    for j in range(128,np.shape(l_on_nir)[0]):
        if l_on_nir[j] >= stop_intensity:
            stop_index = j-1
            break

    plt.figure()
    plt.plot(l_on_vis)
    plt.plot(l_on_nir)
    plt.show()

    l_on_nir = l_on_nir[start_index:stop_index]
    print('Case 2 : ' , start_index , stop_index)
    c_on_nir = [d_on_nir['p_' + str(seek_pixels_set_1[k]) + '_b_' + str(wl_set[14])][start_index:stop_index] for k in range(0,3)]
    ###


    for i in range(0,len(seek_pixels_set_1)):

        plt.figure()
        title = '610nm , calibration curves , pixel' + ' ' + str(seek_pixels_set_1[i])
        save_name = 'b_610_p_' + str(seek_pixels_set_1[i]) + 'masks.png'
        plt.title(title)
        plt.ylim([256,4096])
        plt.plot(l_off_vis,c_off_vis[i])
        plt.plot(l_on_vis,c_on_vis[i])
        plt.legend([' no mask , no blocker' , 'yes mask , no blocker '])
        plt.savefig(save_name)
        #plt.show()

        plt.figure()
        title = '930nm , calibration curves , pixel' + ' ' + str(seek_pixels_set_1[i])
        save_name = 'b_930_p_' + str(seek_pixels_set_1[i]) + 'masks.png'
        plt.title(title)
        plt.ylim([256,4096])
        plt.plot(l_off_nir,c_off_nir[i])
        plt.plot(l_on_nir,c_on_nir[i])
        plt.legend(['no mask , no blocker' , 'yes mask , no blocker'])
        plt.savefig(save_name)
        #plt.show()