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
stop_intensity  = 2400

if __name__ == '__main__' : 

    print('')
    root            = os.getcwd() 
    parent          = Path(root)
    parent          = parent.parent.absolute()

    calib_data_path = os.path.join(parent,'x15_calib')
    calib_data_path = os.path.join(calib_data_path,'2022517_x15_calib')
    calib_children  = glob.glob(os.path.join(calib_data_path,'*.csv'))

    agg_dict        = {}
    N               = len(calib_children)
    print('N files : ' , N)

    for i in range(0,N):
        descript = process_x15_rt_file_name(calib_children[i])

        tool             = descript['tool']
        case             = calib_children[i].split('/')[-1].split('_led_')[-1].split('.csv')[0]

        if case == '6' and tool == 'x15_ud1':

            rep_led  = int(case[0]) - 1

            datagram = pd.read_csv(calib_children[i]).values
            n        = np.shape(datagram)[0]

            leds     = datagram[:,0:n_control]
            cmos     = datagram[:,n_control:n_control + (n_wls * n_pixels)]

            ls        = leds[:,rep_led]
            ind_start , ind_end = 0 , 0
            for j in range(0,np.shape(ls)[0]):
                if ls[j] > start_intensity:
                    ind_start = j
                    break
            
            for j in range(0,np.shape(ls)[0]):
                if ls[j] > stop_intensity:
                    ind_end = j
                    break

            print(ind_start , ind_end)

            look_at = 72

            title = 'p_' + str(look_at) + '_b_' + str(wl_set[14]) + '_t_' + tool
            band  = np.zeros([ind_end - ind_start + 1])
            for l in range(ind_start,ind_end):
                band[l-ind_start] = (divide(cmos[l,:]))[14,look_at]

            plt.figure()
            plt.title(title)
            plt.ylim([256,4096])
            plt.xlabel('led intensity')
            plt.ylabel('cmos count')
            plt.plot(ls[ind_start:ind_end],band[0:-1])
            plt.show()
