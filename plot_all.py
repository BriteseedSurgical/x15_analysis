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

tissue_pick_config =  [['ureter','none'],['peritoneum','none']]

sensor_floor        = 425

watch_baseline        = 1600
plot_baseline         = False
show_legend           = False
plot_norm             = False
averaging_patch_width = 20
homo_avg_start        = 32
homo_avg_end          = 48
vertical_lines        = 0

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

    skeletal_tissue_mean_levels_path = os.path.join(x15_rt_processed_path,'x15_sk_mean_signal_levels' + vertical_line_message)
    if not os.path.exists(skeletal_tissue_mean_levels_path):
        os.mkdir(skeletal_tissue_mean_levels_path)

    for i in range(0,n_children):
        print('Processing , Child : ' , children[i])
        #create if doesn't exist , folder in the top processed folder.
        
        child_name              = children[i].split('/')[-1] 

        child_tag = child_name.split('_rt')[-1]
        print('Tag : ' , len(child_tag) , child_tag)

        sk_mean_levels_file_name = 'x15_sk_mean_levels_b_' + str(watch_baseline) + '_' + child_name
        nu_mean_levels_file_name = 'x15_nu_mean_levels_b_' + str(watch_baseline) + '_' + child_name
        plot_destination         = sk_mean_levels_file_name

        targets                 = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets               = len(targets)
        targets_crossed         = targets_crossed + n_targets
        print('Targets in path : ' , n_targets)

        meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
        meta_keys               = meta_keys['keys']

        skeletal_ureter_list = []
        skeletal_peri_list   = []
        native_ure_list      = []
        reference_target     = ''
        reference_control    = None
        reference_read       = None

        reference_found = False
        for j in range(0,n_targets):
            meta = process_x15_rt_file_name(targets[j])
            if meta['baseline'] == str(watch_baseline) and meta['vessel'] == 'none' and meta['tissue'] == 'none' and meta['comment'] == 'dailytest':
                reference_target = targets[j]
                print('Reference file found and stored.')
                reference_found = True
                break

        if reference_found:
            reference_content = process_file_content_x15_rt_two_step(reference_target)
            reference_control = reference_content['l']
            reference_read    = reference_content['x15_c']        

        for j in range(0,n_targets):
            meta = process_x15_rt_file_name(targets[j])
            if meta['baseline'] == str(watch_baseline):
                if meta['vessel'] == 'ureter'     and meta['tissue'] == 'none' and meta['position_1'] == '1-99':
                    skeletal_ureter_list.append(targets[j])
                if meta['vessel'] == 'peritoneum' and meta['tissue'] == 'none' and meta['position_1'] == '1-99':
                    skeletal_peri_list.append(targets[j])
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
                    #for native ureters , average in min (20pixels , labelled ROI) 
                    if covered:
                        native_ure_list.append(targets[j])

        print('Skeletal tissue counts : ' , len(skeletal_ureter_list) , len(skeletal_peri_list))
        print('Native ureter counts   : ' , len(native_ure_list))
        
        ure_save_at       = os.path.join(skeletal_tissue_mean_levels_path,sk_mean_levels_file_name + '_ure.pdf')
        peri_save_at      = os.path.join(skeletal_tissue_mean_levels_path,sk_mean_levels_file_name + '_peri.pdf')
        nu_save_at        = os.path.join(skeletal_tissue_mean_levels_path,nu_mean_levels_file_name + '_nu.pdf')

        ure_norm_save_at  = os.path.join(skeletal_tissue_mean_levels_path,sk_mean_levels_file_name + '_ure_norm.pdf')
        peri_norm_save_at = os.path.join(skeletal_tissue_mean_levels_path,sk_mean_levels_file_name + '_peri_norm.pdf')

        #Plot and save skeletal ureters.
        if len(skeletal_ureter_list) > 0:
            plt.figure()
            plt.xlabel('wavelengths (nm)')
            plt.ylabel('mean cmos count')
            plt.title('Tx intensities - Ureters')
            plt.ylim([256,4096])
            legend = []
            for j in range(0,len(skeletal_ureter_list)):
                current_meta    = process_x15_rt_file_name(skeletal_ureter_list[j])
                current_content = process_file_content_x15_rt_two_step(skeletal_ureter_list[j])
                legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                plt.plot(wl_set,np.mean(current_content['x15_c'],axis=1))
            if plot_baseline:
                legend.append('baseline')
                plt.plot(wl_set,np.ones([n_wls]) * watch_baseline)
            if vertical_lines:
                for j in range(0,len(wl_set)):
                    plt.axvline(wl_set[j])
            if show_legend:
                plt.legend(legend)
            plt.savefig(ure_save_at)
             
            if reference_found and plot_norm:
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title('Tx intensities - Ureters , normalized')
                plt.ylim([0,1])
                legend = []
                for j in range(0,len(skeletal_ureter_list)):
                    current_meta       = process_x15_rt_file_name(skeletal_ureter_list[j])
                    current_content    = process_file_content_x15_rt_two_step(skeletal_ureter_list[j])
                    current_normalized = np.divide(current_content['x15_c'],reference_read)
                    current_normalized = current_normalized/np.max(current_normalized)
                    legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                    plt.plot(wl_set,np.mean(current_normalized,axis=1))
                if vertical_lines:
                    for j in range(0,len(wl_set)):
                        plt.axvline(wl_set[j])
                #plt.legend(legend)
                plt.savefig(ure_norm_save_at)
        #Plot and save skeletal ureters.

        #Plot and save native ureters.
        if len(native_ure_list) > 0:
            plt.figure()
            plt.xlabel('wavelengths (nm)')
            plt.ylabel('mean cmos count')
            plt.title('Tx intensities - Native Ureters')
            plt.ylim([256,4096])
            legend = []
            for j in range(0,len(native_ure_list)):
                current_meta    = process_x15_rt_file_name(native_ure_list[j])
                current_content = process_file_content_x15_rt_two_step(native_ure_list[j])
                legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                plt.plot(wl_set,np.mean(current_content['x15_c'],axis=1))
            if plot_baseline:
                legend.append('baseline')
                plt.plot(wl_set,np.ones([n_wls]) * watch_baseline)
            if vertical_lines:
                for j in range(0,len(wl_set)):
                    plt.axvline(wl_set[j])
            if show_legend:
                plt.legend(legend)
            plt.savefig(nu_save_at)
        #Plot and save native ureters.


        #Plot and save skeletal Peritoneums.
        if len(skeletal_peri_list) > 0:

            plt.figure()
            plt.xlabel('wavelengths (nm)')
            plt.ylabel('mean cmos count')
            plt.title('Tx intensities - Peritoneums')
            plt.ylim([256,4096])
            legend = []
            for j in range(0,len(skeletal_peri_list)):
                current_meta    = process_x15_rt_file_name(skeletal_peri_list[j])
                current_content = process_file_content_x15_rt_two_step(skeletal_peri_list[j])
                legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                plt.plot(wl_set,np.mean(current_content['x15_c'],axis=1))
            if plot_baseline:
                legend.append('baseline')
                plt.plot(wl_set,np.ones([n_wls]) * watch_baseline)
            if vertical_lines:
                for j in range(0,len(wl_set)):
                    plt.axvline(wl_set[j])
            if show_legend:
                plt.legend(legend)
            plt.savefig(peri_save_at)      
            
            if reference_found and plot_norm:
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title('Tx intensities - Peritoneums , normalized')
                plt.ylim([0,1])
                legend = []
                for j in range(0,len(skeletal_peri_list)):
                    current_meta       = process_x15_rt_file_name(skeletal_peri_list[j])
                    current_content    = process_file_content_x15_rt_two_step(skeletal_peri_list[j])
                    current_normalized = np.divide(current_content['x15_c'],reference_read)
                    current_normalized = current_normalized/np.max(current_normalized)
                    legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                    plt.plot(wl_set,np.mean(current_normalized,axis=1))
                if vertical_lines:
                    for j in range(0,len(wl_set)):
                        plt.axvline(wl_set[j])
                plt.legend(legend)
                plt.savefig(peri_norm_save_at)