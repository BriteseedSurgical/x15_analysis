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

#lambdas
vis_read = lambda x: (int(x[1] + x[3] + x[5]) == int(48))
nir_read = lambda x: (int(x[0] + x[2] + x[4]) == int(48))
#lambdas

#calib-adapt pixels
pixels_vis_calib_adapt = [4,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

tissue_exclude_list = ['none']
sensor_floor        = 425

watch_baseline      = 1600
vertical_lines      = 1


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
    n_children      = len(children)
    targets_crossed = 0

    x15_rt_processed_path    = os.path.join(parent,'x15_rt_processed')
    if not os.path.exists(x15_rt_processed_path):
        os.mkdir(x15_rt_processed_path)

    if not vertical_lines:
        vertical_line_message = '_no_vlines'
    else:
        vertical_line_message = ''

    mean_levels_path = os.path.join(x15_rt_processed_path,'x15_mean_signal_levels' + vertical_line_message)
    if not os.path.exists(mean_levels_path):
        os.mkdir(mean_levels_path)

    for i in range(0,n_children):
        print('Processing , Child : ' , children[i])
        #create if doesn't exist , folder in the top processed folder.
        
        child_name              = children[i].split('/')[-1] 

        child_tag = child_name.split('_rt')[-1]
        print('Tag : ' , len(child_tag) , child_tag)

        mean_levels_folder_name = 'x15_mean_levels_b_' + str(watch_baseline) + '_' + child_name
        
        plots_destination       = os.path.join(mean_levels_path,mean_levels_folder_name)
        if not os.path.exists(plots_destination):
            os.mkdir(plots_destination)

        targets                 = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets               = len(targets)
        targets_crossed         = targets_crossed + n_targets
        print('Targets in path : ' , n_targets)

        meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
        meta_keys               = meta_keys['keys']

        for j in range(0,n_targets):

            #print('File target : ' , targets[j])
            target_name      = targets[j].split('/')[-1].split('.csv')[0]
            meta             = process_x15_rt_file_name(targets[j])

            position1 = meta['position_1']
            p1_start , p1_end  = int(position1.split('-')[0]) , int(position1.split('-')[-1])
            position2 = meta['position_2']
            p2_start , p2_end  = int(position2.split('-')[0]) , int(position2.split('-')[-1])
            min_start , max_end = np.min([p1_start,p1_end]) , np.max([p1_end,p2_end])
            if min_start == 1 and max_end == 99:
                covered = True
            else:
                covered = False

            if covered:#meta['baseline'] == str(watch_baseline):# and meta['comment'] == 'dailytest' :#and covered:#and not (meta['tissue'] in tissue_exclude_list and meta['vessel'] in tissue_exclude_list):
                
                print(targets[j])
                processed        = process_file_content_x15_rt_two_step(targets[j])
                
                if not processed['incomplete_data']:
    
                    final_state   = processed['x15_c']
                    final_control = processed['l']

                    current_plot_title = 'x15_mean_levels. Primary RoI - ' + str(p1_start) + '-' + str(p1_end) + ' leds at ' + str(final_control)
                    current_plot_dest  = os.path.join(plots_destination,target_name + '.pdf')

                    if meta['vessel'] == 'ureter' and meta['tissue'] == 'none' and meta['baseline'] == str(watch_baseline):
                        plot_x15_spatial(final_state , baseline = 'Test', save_to = current_plot_title)
                        exit(0)

                        plot_mean_levels(final_state,p1_start,p1_end,int(meta['baseline']),current_plot_title,\
                                     current_plot_dest,vert = vertical_lines,save = True,show = False)

                        plot_mean_levels(final_state,min(pixels_nir_calib_adapt),max(pixels_vis_calib_adapt),int(meta['baseline']),current_plot_title,\
                                     current_plot_dest,vert = vertical_lines,save = True,show = False)

                    #plot_x15_adapt(processed['x15_c'][4,:],processed['x15_c'][14,:],watch_baseline,save = False,  show = True)


    print('Total targets crossed : ' , targets_crossed)
    print('At main exit line.')