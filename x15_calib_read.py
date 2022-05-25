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

start_intensity = 400
stop_intensity  = 3800

seek_leds_set_2 = [500,700,1000,1500,2000,2500]


if __name__ == '__main__' : 

    print('')
    root            = os.getcwd() 
    parent          = Path(root)
    parent          = parent.parent.absolute()
    
    calib_data_path = os.path.join(parent,'x15_calib_masks_and_blockers')
    calib_children  = glob.glob(os.path.join(calib_data_path,'*'))

    agg_dict        = {}
    N               = len(calib_children)
    
    exec_no_blocker = 0 

    for i in range(0,N):
        descript = calib_children[i].split('/')[-1].split('x15_calib_')[-1]

        line             = descript.split('_mask')[0]
        mask             = int(descript.split('mask_')[-1].split('_')[0])
        blocker          = int(descript.split('blocker_')[-1].split('_')[0])
        blocker_position = descript.split('_ja_')[0].split('_')[-1]

        if blocker == 1 and blocker_position == 'centre':

            d     = read_calib_data_discrete_leds(calib_children[i],seek_leds_set_2)
            d_vis = d['135']
            d_nir = d['246']
    #plots
    #6(LED Intensities) * 2(WL,NIR) plots , each comparing spatial curve when 
    #WL or NIR LEDs on at each of the target LEDs.

    plt.figure()
    title = 'Spatial response 610nm with LEDs 1,3,5 ON'
    plt.title(title)
    plt.xlabel('pixel count')
    plt.ylabel('cmos intensity')
    plt.ylim([256,4096])
    for i in range(0,len(seek_leds_set_2)):
        sig_vis  = d_vis[str(seek_leds_set_2[i])]
        sig_vis  = sig_vis[4,:]
        plt.plot(sig_vis)
    legend = ['Intensity : ' + str(seek_leds_set_2[k]) for k in range(0,len(seek_leds_set_2))]
    plt.legend(legend)
    plt.savefig('spatial_response_discrete_leds_610nm')
    plt.show()

    plt.figure()
    title = 'Spatial response 930 nm with LEDs 2,4,6 ON'
    plt.title(title)
    plt.xlabel('pixel count')
    plt.ylabel('cmos intensity')
    plt.ylim([256,4096])
    for i in range(0,len(seek_leds_set_2)):
        sig_nir  = d_nir[str(seek_leds_set_2[i])]
        sig_nir  = sig_nir[14,:]
        plt.plot(sig_nir)
    legend = ['Intensity : ' + str(seek_leds_set_2[k]) for k in range(0,len(seek_leds_set_2))]
    plt.legend(legend)
    plt.savefig('spatial_response_discrete_leds_930nm')
    plt.show()



    if 1:
        for i in range(0,N):
            descript = calib_children[i].split('/')[-1].split('x15_calib_')[-1]
            #print('Descript : ' , descript)

            #experiment 1
            line    = descript.split('_mask')[0]
            mask    = int(descript.split('mask_')[-1].split('_')[0])
            blocker = int(descript.split('blocker_')[-1].split('_')[0])

            if blocker == 0 and line == 'all_lines':
                print('Current folder :' , descript)
                exec_no_blocker = exec_no_blocker + 1
                if not mask:
                    calib_no_mask  = read_calib_data(calib_children[i],4,[12,50,80],14,[12,50,80])
                else:
                    calib_yes_mask = read_calib_data(calib_children[i],4,[12,50,80],14,[12,50,80])

            else:
                exec_no_blocker = exec_no_blocker - 1
                if blocker == 1:
                    print('Current folder : ' , descript)
                    blocker_position = descript.split('_ja_')[0].split('_')[-1]

                    if blocker_position == 'centre':
                        
                        d     = read_calib_data(calib_children[i],4,[12,50,80],14,[12,50,80])
                        
                        d_vis = d['135']
                        l_vis = d_vis['l']

                        start_index = 0
                        stop_index  = 0
                        for j in range(0,np.shape(l_vis)[0]):
                            if l_vis[j] >= start_intensity:
                                start_index = j
                                break
                        for j in range(0,np.shape(l_vis)[0]):
                            if l_vis[j] >= stop_intensity:
                                stop_index = j-1
                                break
                        print(start_index,stop_index)
                        l_vis = l_vis[start_index:stop_index]
                        c_vis = [d_vis['p_' + str([12,50,80][k]) + '_b_' + str(wl_set[4])][start_index:stop_index] for k in range(0,3)]

                        
                        plt.figure()
                        plt.xlabel('LED Intensity')
                        plt.ylabel('CMOS Intensity')
                        title = 'Response of 610nm to LEDs 1,3,5 - Blocker at pixels 40-60'
                        save_name = 'blocker_front_response_610nm_L_135' + '.png'
                        plt.ylim([256,4096])
                        plt.title(title)
                        plt.plot(l_vis,c_vis[0])
                        plt.plot(l_vis,c_vis[1])
                        plt.plot(l_vis,c_vis[2])
                        plt.legend(['pixel 12' , 'pixel 50' , 'pixel 80'])
                        plt.savefig(save_name)
                        plt.show()
                        

                        d_nir = d['246']
                        l_nir = d_nir['l']

                        start_index = 0
                        stop_index  = 0
                        for j in range(0,np.shape(l_nir)[0]):
                            if l_nir[j] >= start_intensity:
                                start_index = j
                                break
                        for j in range(0,np.shape(l_nir)[0]):
                            if l_nir[j] >= stop_intensity:
                                stop_index = j-1
                                break
                        print(start_index,stop_index)
                        l_nir = l_nir[start_index:stop_index]
                        c_nir = [d_nir['p_' + str([12,50,80][k]) + '_b_' + str(wl_set[14])][start_index:stop_index] for k in range(0,3)]

                        
                        plt.figure()
                        plt.xlabel('LED Intensity')
                        plt.ylabel('CMOS Intensity')
                        title = 'Response of 930nm to LEDs 2,4,6 - Blocker at pixels 40-60'
                        save_name = 'blocker_front_response_930nm_L_246' + '.png'
                        plt.ylim([256,4096])
                        plt.title(title)
                        plt.plot(l_nir,c_nir[0])
                        plt.plot(l_nir,c_nir[1])
                        plt.plot(l_nir,c_nir[2])
                        plt.legend(['pixel 24' , 'pixel 50' , 'pixel 80'])
                        plt.savefig(save_name)
                        plt.show()
                        

                    if blocker_position == 'centre':
                        _ = 0

                    if blocker_position == 'front':
                        _ = 0

    """
    no_mask_case_135       = calib_no_mask['135']
    no_mask_vis_leds       = no_mask_case_135['l']
    start_index = 0
    stop_index  = 0
    for i in range(0,np.shape(no_mask_vis_leds)[0]):
        if no_mask_vis_leds[i] >= start_intensity:
            start_index = i
            break
    for i in range(0,np.shape(no_mask_vis_leds)[0]):
        if no_mask_vis_leds[i] >= stop_intensity:
            stop_index = i-1
            break
    no_mask_vis_leds       = no_mask_vis_leds[start_index:stop_index]
    no_mask_vis_cmos       = [no_mask_case_135['p_' + str([12,50,80][j]) + '_b_' + str(wl_set[4])][start_index:stop_index] for j in range(0,len([12,50,80])) ]




    no_mask_case_246       = calib_no_mask['246']
    no_mask_nir_leds       = no_mask_case_246['l']
    start_index = 0
    stop_index  = 0
    for i in range(0,np.shape(no_mask_nir_leds)[0]):
        if no_mask_nir_leds[i] >= start_intensity:
            start_index = i
            break
    for i in range(0,np.shape(no_mask_nir_leds)[0]):
        if no_mask_nir_leds[i] >= stop_intensity:
            stop_index = i-1
            break
    no_mask_nir_leds       = no_mask_nir_leds[start_index:stop_index]
    no_mask_nir_cmos       = [no_mask_case_246['p_' + str([12,50,80][j]) + '_b_' + str(wl_set[n_wls-1])][start_index:stop_index] for j in range(0,len([12,50,80]))]




    yes_mask_case_135       = calib_yes_mask['135']
    yes_mask_vis_leds       = yes_mask_case_135['l']
    start_index = 0
    stop_index  = 0
    for i in range(0,np.shape(yes_mask_vis_leds)[0]):
        if yes_mask_vis_leds[i] >= start_intensity:
            start_index = i
            break
    for i in range(0,np.shape(yes_mask_vis_leds)[0]):
        if yes_mask_vis_leds[i] >= stop_intensity:
            stop_index = i-1
            break
    yes_mask_vis_leds       = yes_mask_vis_leds[start_index:stop_index]
    yes_mask_vis_cmos       = [yes_mask_case_135['p_' + str([12,50,80][j]) + '_b_' + str(wl_set[4])][start_index:stop_index] for j in range(0,len([12,50,80])) ]




    yes_mask_case_246       = calib_yes_mask['246']
    yes_mask_nir_leds       = yes_mask_case_246['l']
    start_index = 0
    stop_index  = 0
    for i in range(0,np.shape(yes_mask_nir_leds)[0]):
        if yes_mask_nir_leds[i] >= start_intensity:
            start_index = i
            break
    for i in range(0,np.shape(yes_mask_nir_leds)[0]):
        if yes_mask_nir_leds[i] >= stop_intensity:
            stop_index = i-1
            break
    yes_mask_nir_leds       = yes_mask_nir_leds[start_index:stop_index]
    yes_mask_nir_cmos       = [yes_mask_case_246['p_' + str([12,50,80][j]) + '_b_' + str(wl_set[n_wls-1])][start_index:stop_index] for j in range(0,len([12,50,80]))]        

    plt.figure()
    plt.xlabel('LED Intensity')
    plt.ylabel('CMOS Intensity')
    title = 'Response of pixel 50 of 610nm to LEDs 1,3,5'
    save_name = 'p_50_response_610nm_L_135' + '.png'
    plt.ylim([256,4096])
    plt.title(title)
    plt.plot(no_mask_vis_leds,no_mask_vis_cmos[1])
    plt.plot(yes_mask_vis_leds,yes_mask_vis_cmos[1])
    plt.plot(l_vis,c_vis[1])
    plt.legend(['pixel 50 - No mask , No blocker' , 'pixel 50 - Mask on , No blocker' , 'pixel 50 - Mask on , Blocker at pixels 1-24'])
    plt.savefig(save_name)
    plt.show()

    plt.figure()
    plt.xlabel('LED Intensity')
    plt.ylabel('CMOS Intensity')
    title = 'Response of pixel 50 of 930nm to LEDs 2,4,6'
    save_name = 'p_50_response_930nm_L_246' + '.png'
    plt.ylim([256,4096])
    plt.title(title)
    plt.plot(no_mask_nir_leds,no_mask_nir_cmos[1])
    plt.plot(yes_mask_nir_leds,yes_mask_nir_cmos[1])
    plt.plot(l_nir,c_nir[1])
    plt.legend(['pixel 50 - No mask , No blocker' , 'pixel 50 - Mask on , No blocker' , 'pixel 50 - Mask on , Blocker at pixels 1-24'])
    plt.savefig(save_name)
    plt.show()
    """

    """
    for i in range(0,len(pixels_vis_calib_adapt)):

        plt.figure()
        title = 'mask vs no mask of ' + str(wl_set[4]) + ' nm at pixel ' + str(pixels_vis_calib_adapt[i])
        save_name = 'mask_vs_no_mask_of_' + str(wl_set[4]) + '_nm_at_pixel_' + str(pixels_vis_calib_adapt[i]) + '.png'
        plt.title(title)
        plt.xlim([start_intensity , stop_intensity])
        plt.ylim([256 , 4096])
        plt.plot(no_mask_vis_leds,no_mask_vis_cmos[i])
        plt.plot(yes_mask_vis_leds,yes_mask_vis_cmos[i])
        plt.legend(['no mask' , 'mask'])
        plt.savefig(save_name)
        #plt.show()

    for i in range(0,len(pixels_nir_calib_adapt)):

        plt.figure()
        title = 'mask vs no mask of ' + str(wl_set[n_wls-1]) + ' nm at pixel ' + str(pixels_nir_calib_adapt[i])
        save_name = 'mask_vs_no_mask_of_' + str(wl_set[n_wls-1]) + '_nm_at_pixel_' + str(pixels_nir_calib_adapt[i]) + '.png'
        plt.title(title)
        plt.xlim([start_intensity , stop_intensity])
        plt.ylim([256 , 4096])
        plt.plot(no_mask_nir_leds,no_mask_nir_cmos[i])
        plt.plot(yes_mask_nir_leds,yes_mask_nir_cmos[i])
        plt.legend(['no mask' , 'mask'])
        plt.savefig(save_name)
        #plt.show()
    """
    