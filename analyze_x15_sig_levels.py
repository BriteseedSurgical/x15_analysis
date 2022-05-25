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

#default seek indices
seek_leds_set_1 = [200,300,500,700,1000,1500,2000]
#default seek indices

tissue_exclude_list = ['none']
sensor_floor        = 425

watch_baseline      = 1600
track_err_start     = 3
track_err_end       = 3

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
    x15_adapt_processed_path = os.path.join(x15_rt_processed_path,'x15_adapt_analysis')
    if not os.path.exists(x15_adapt_processed_path):
        os.mkdir(x15_adapt_processed_path)
    x15_ref_plots_path       = os.path.join(x15_rt_processed_path,'x15_ref_plots')
    if not os.path.exists(x15_ref_plots_path):
        os.mkdir(x15_ref_plots_path)
    x15_error_plots_save_at  = os.path.join(x15_rt_processed_path,'x15_adapt_error_plots')
    if not os.path.exists(x15_error_plots_save_at):
        os.mkdir(x15_error_plots_save_at)

    for i in range(0,n_children):
        print('Processing , Child : ' , children[i])
        #create if doesn't exist , folder in the top processed folder.
        
        child_name              = children[i].split('/')[-1] 
        signal_levels_file_name = 'x15_analysis_adapt_b_' + str(watch_baseline) + '_' + child_name + '.csv'
        
        error_plots_source      = 'x15_error_plots_b_' + str(watch_baseline) + '_' + child_name
        error_plots_source      = os.path.join(x15_error_plots_save_at,error_plots_source)
        if not os.path.exists(error_plots_source):
            os.mkdir(error_plots_source)

        ref_plots_source = 'x15_ref_plots_b_' + str(watch_baseline) + '_' + child_name
        ref_plots_source = os.path.join(x15_ref_plots_path,ref_plots_source)
        if not os.path.exists(ref_plots_source):
            os.mkdir(ref_plots_source)

        targets                 = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets               = len(targets)
        targets_crossed         = targets_crossed + n_targets
        print('Targets in path : ' , n_targets)

        meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
        meta_keys               = meta_keys['keys']

        processed_file_header   = []
        for j in range(0,len(meta_keys)):
            processed_file_header.append(meta_keys[j])

        processed_file_header.append('n_cycles')
        processed_file_header.append('mean(' + str(wl_set[4]) + ')')
        processed_file_header.append('mean(' + str(wl_set[14]) + ')')
        for j in range(0,len(pixels_vis_calib_adapt)):
            processed_file_header.append('d_vis_' + str(pixels_vis_calib_adapt[j]))
        for j in range(0,len(pixels_nir_calib_adapt)):
            processed_file_header.append('d_nir_' + str(pixels_nir_calib_adapt[j]))
        processed_file_header.append('rt_l1_vis')
        processed_file_header.append('rt_l1_nir')
        processed_file_header.append('vis_shift')
        processed_file_header.append('nir_shift')
        processed_file_header.append('vis_dev_l1')
        processed_file_header.append('nir_dev_l1')
        processed_file_header.append('0.9x_vis')
        processed_file_header.append('0.9x_nir')
        processed_file_header.append('0.8x_vis')
        processed_file_header.append('0.8x_nir')
        processed_file_header.append('converged')
        for j in range(0,n_control):
            processed_file_header.append('LED ' + str(j+1))

        file_pointer = open(os.path.join(x15_adapt_processed_path,signal_levels_file_name),'w',encoding='UTF8')
        file_writer  = csv.writer(file_pointer)
        file_writer.writerow(processed_file_header)
        
        for j in range(0,n_targets):

            print('File target : ' , targets[j])
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

            if meta['baseline'] == str(watch_baseline) and not (meta['tissue'] in tissue_exclude_list and meta['vessel'] in tissue_exclude_list) and covered:
                
                processed        = process_file_content_x15_rt_two_step(targets[j])
                
                
                if not processed['incomplete_data']:
    
                    ref_plot_save_at = targets[j].split('/')[-1].split('.csv')[0] + '.png'

                    analysis         = analyze_adapt(processed['x15_cs'],b = watch_baseline , index_vis = 4 , index_nir = 14)
                    convergence      = processed['converged']
                    final_state      = processed['l']

                    plot_x15_adapt(processed['x15_c'][4,:],processed['x15_c'][14,:],watch_baseline,save_at=os.path.join(ref_plots_source,ref_plot_save_at))
                    
                    content_to_file  = []
                    for k in range(0,len(meta['keys'])):
                        content_to_file.append(meta[meta['keys'][k]])

                    content_to_file.append(analysis['n_cycles'])
                    content_to_file.append(analysis['mean_vis'])
                    content_to_file.append(analysis['mean_nir'])

                    for l in range(0,len(pixels_vis_calib_adapt)):
                        content_to_file.append(analysis['d_vis_end'][l])
                    for l in range(0,len(pixels_nir_calib_adapt)):
                        content_to_file.append(analysis['d_nir_end'][l])

                    content_to_file.append(analysis['err_rt_vis'])
                    content_to_file.append(analysis['err_rt_nir'])

                    content_to_file.append(analysis['vis_shift'])
                    content_to_file.append(analysis['nir_shift'])

                    content_to_file.append(analysis['vis_dev_l1'])
                    content_to_file.append(analysis['nir_dev_l1'])

                    #N1,N2,N3,N4 , plot error curves
                    vis_errors = analysis['l1_vis_errors']
                    nir_errors = analysis['l1_nir_errors']

                    if len(vis_errors) == 0 or len(nir_errors) == 0:
                        print('Shape check fail.')
                    else:
                        #print(type(vis_errors))
                        vis_errors = vis_errors[::-1]
                        nir_errors = nir_errors[::-1]

                    if analysis['n_cycles'] > track_err_start + track_err_end + 1:
                        vis_errors_central = vis_errors[track_err_start:len(vis_errors) - track_err_end]
                        nir_errors_central = nir_errors[track_err_start:len(nir_errors) - track_err_end]
                    else:
                        vis_errors_central = vis_errors
                        nir_errors_central = nir_errors

                    #provably monotonic ? 
                    #few files have spikes of error at the end.
                    vis_error_adjusted = vis_errors_central[0] - vis_errors_central[-1]
                    nir_error_adjusted = nir_errors_central[0] - nir_errors_central[-1]
                    
                    threshold_vis_x    = 0.9 * vis_error_adjusted
                    threshold_vis_y    = 0.8 * vis_error_adjusted
                    
                    threshold_nir_x    = 0.9 * nir_error_adjusted
                    threshold_nir_y    = 0.8 * nir_error_adjusted

                    x_vis_break_point  = vis_errors_central[0] - threshold_vis_x
                    x_nir_break_point  = nir_errors_central[0] - threshold_nir_x

                    y_vis_break_point  = vis_errors_central[0] - threshold_vis_y
                    y_nir_break_point  = nir_errors_central[0] - threshold_nir_y

                    N1 , N2 , N3 , N4 = -1,-1,-1,-1

                    ctr1 = 0
                    ctr2 = 0
                    for l in range(0,len(vis_errors_central)):
                        if verify_residual_slope(vis_errors_central , l , x_vis_break_point):
                            ctr1 = l
                            break   
                    for l in range(0,len(nir_errors_central)):
                        if verify_residual_slope(nir_errors_central , l , x_nir_break_point):
                            ctr2 = l
                            break 
                    N1 = ctr1 + track_err_start
                    N2 = ctr2 + track_err_start   

                    ctr1 = 0
                    ctr2 = 0
                    for l in range(0,len(vis_errors_central)):
                        if verify_residual_slope(vis_errors_central , l , y_vis_break_point):
                            ctr1 = l
                            break   
                    for l in range(0,len(nir_errors_central)):
                        if verify_residual_slope(nir_errors_central , l , y_nir_break_point):
                            ctr2 = l
                            break 
                    N3 = ctr1 + track_err_start
                    N4 = ctr2 + track_err_start  

                    plt.figure()
                    plt.title('Error vs adapt time')
                    plt.xlabel('adapt step')
                    plt.ylabel('L1(ref[pixel i] - baseline)')
                    plt.xlim([1,analysis['n_cycles']])
                    plt.plot(vis_errors,'b-')
                    plt.plot(nir_errors,'r-')
                    plt.axvline(N1,color='b')
                    plt.axvline(N2,color='r')
                    plt.legend(['vis erros vs time','nir errors vs time','vis 0.9x solution point','nir 0.9x solution point'])
                    error_plots_save_at = targets[j].split('/')[-1].split('.csv')[0] + '.png'
                    error_plots_save_at = os.path.join(error_plots_source,error_plots_save_at)
                    plt.savefig(error_plots_save_at)
                    #plt.show()

                    content_to_file.append(N1)
                    content_to_file.append(N2)
                    content_to_file.append(N3)
                    content_to_file.append(N4)

                    content_to_file.append(convergence)
                    for k in range(0,n_control):
                        content_to_file.append(final_state[k])

                    file_writer.writerow(content_to_file)
    
    print('')

    print('Targets crossed : ' , targets_crossed)

    print('')
    print('At main exit line.')