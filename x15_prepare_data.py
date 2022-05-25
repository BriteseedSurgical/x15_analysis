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

tissue_pick_config =  [['ureter','none'],['peritoneum','none']]

sensor_floor        = 425

watch_baseline      = 1600
vertical_lines      = 0

reference_keywords  = ['dailytest' , 'daily_test']

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

    x15_rt_processed_path = os.path.join(x15_rt_processed_path , 'x15_msls')
    if not os.path.exists(x15_rt_processed_path):
        os.mkdir(x15_rt_processed_path)


    for i in range(0,n_children):
        
        child_name              = children[i].split('/')[-1] 
        print('')
        print('Current child : ' , child_name)

        targets                 = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets               = len(targets)

        meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
        meta_keys               = meta_keys['keys']

        unique_configurations   = []

        categorized_lists       = {}

        for j in range(0,n_targets):

            current_meta      = process_x15_rt_file_name(targets[j])
            current_jaw_angle = current_meta['ja_manual'] #a string.
            current_pig_id    = current_meta['pig1']      #a string.

            current_configuration = 'ja_' + current_jaw_angle + '_pig_' + current_pig_id
            if current_configuration not in unique_configurations:
                unique_configurations.append(current_configuration)
        
        print('Unique configurations.')
        print(unique_configurations)
        print('')

        for j in range(0,len(unique_configurations)):

            subset = []
            for k in range(0,n_targets):

                current_meta      = process_x15_rt_file_name(targets[k])
                current_jaw_angle = current_meta['ja_manual'] #a string.
                current_pig_id    = current_meta['pig1']      #a string.

                current_configuration = 'ja_' + current_jaw_angle + '_pig_' + current_pig_id
                if current_configuration == unique_configurations[j] and current_meta['baseline'] == str(watch_baseline):
                    subset.append(targets[k])

            categorized_lists[unique_configurations[j]] = subset

        #verification.
        print('')
        for key,item in categorized_lists.items():
            print('Case : ' , key)
            print('N    : ' , len(categorized_lists[key]))
        #verification.

        #Current date's main folder.
        main_folder_name = os.path.join(x15_rt_processed_path , child_name)
        if not os.path.exists(main_folder_name):
            os.mkdir(main_folder_name)
        #Current date's main folder.
        
        #Extract air-reference information. Can better set this up. Currently too rigid.
        reference_20_found  = False
        reference_30_found  = False
        reference_20_target = None
        reference_30_target = None

        for j in range(0,n_targets):
            meta = process_x15_rt_file_name(targets[j])
            if meta['baseline'] == str(watch_baseline) and \
               meta['vessel'] == 'none' and meta['tissue'] == 'none' and \
               (meta['comment'] in reference_keywords) and \
               meta['ja_manual'] == '20':
                
                reference_20_target = targets[j]
                print('Reference file found and stored for ja = 20.')
                reference_20_found = True
                break

        for j in range(0,n_targets):
            meta = process_x15_rt_file_name(targets[j])
            if meta['baseline'] == str(watch_baseline) and \
               meta['vessel'] == 'none' and meta['tissue'] == 'none' and \
               (meta['comment'] in reference_keywords) and \
               meta['ja_manual'] == '30':
                
                reference_30_target = targets[j]
                print('Reference file found and stored for ja = 30.')
                reference_30_found = True
                break
        
        if reference_20_found and reference_30_found:

            reference_20_content = process_file_content_x15_rt_two_step(reference_20_target)
            reference_20_control = reference_20_content['l']
            reference_20_read    = reference_20_content['x15_c'] 

            reference_30_content = process_file_content_x15_rt_two_step(reference_30_target)
            reference_30_control = reference_30_content['l']
            reference_30_read    = reference_30_content['x15_c'] 
        #Extract air-reference information. Can better set this up. Currently too rigid.
        
        for j in range(0,len(unique_configurations)):

            #Make sub folder for each config 
            config_folder = os.path.join(main_folder_name , unique_configurations[j])
            if not os.path.exists(config_folder):
                os.mkdir(config_folder)
            #Make sub folder for each config
        
            #Raw and normalized subfolders inside each config folder.
            raw_plots_folder = os.path.join(config_folder,'raw')
            if not os.path.exists(raw_plots_folder):
                os.mkdir(raw_plots_folder)
            
            air_ref_plots_folder = os.path.join(config_folder , 'air_ref_normalized')
            if not os.path.exists(air_ref_plots_folder):
                os.mkdir(air_ref_plots_folder)
            #Raw and normalized subfolders inside each config folder.

            selected_config    = unique_configurations[j]

            selected_pig_id    = selected_config.split('_pig_')[-1]
            selected_jaw_angle = selected_config.split('ja_')[-1].split('_')[0]

            #Find ureters and peritoneums in this sublist.
            
            native_ureter_sublist   = []
            native_ureter_rois      = [] 

            skeletal_ureter_sublist = []
            skeletal_ureter_rois    = [] 
            
            peritoneum_sublist      = []
            peritoneum_rois         = []

            relevant_files = categorized_lists[selected_config]
            for k in range(0,len(relevant_files)):
                
                meta = process_x15_rt_file_name(relevant_files[k])
                
                roi  = meta['position_1']
                roi_start = int(roi.split('-')[0])
                roi_end   = int(roi.split('-')[-1])
                if roi_start < 0:
                    roi_start = 0
                if roi_end > n_pixels:
                    roi_end = n_pixels
                roi_centre = int(0.5*(roi_start + roi_end))

                current_roi = [roi_start , roi_end , roi_centre]

                # I am ignoring the check to see if the entire sensor is covered. 
                if meta['vessel'] == 'ureter' and meta['tissue'] == 'none':
                    skeletal_ureter_sublist.append(relevant_files[k])
                    skeletal_ureter_rois.append(current_roi)

                if meta['vessel'] == 'peritoneum' and meta['tissue'] == 'none':
                    peritoneum_sublist.append(relevant_files[k])
                    peritoneum_rois.append(current_roi)

                if meta['vessel'] == 'ureter' and meta['tissue'] == 'peritoneum':
                    native_ureter_sublist.append(relevant_files[k])
                    native_ureter_rois.append(current_roi)
                # I am ignoring the check to see if the entire sensor is covered.
            #Find ureters and peritoneums in this sublist.
            
            print('N native ureters : ' , len(native_ureter_sublist))
            print('N skeletal peris : ' , len(peritoneum_sublist))
            print('N skeletal ures  : ' , len(skeletal_ureter_sublist))
            
            if selected_jaw_angle == '20':
                reference_read = reference_20_read
            if selected_jaw_angle == '30':
                reference_read = reference_30_read

            #Plot skeletal ureters.
            if len(skeletal_ureter_sublist) > 0:
                
                #plot raw signal levels.
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title( 'Tx intensities - Ureters')
                plt.ylim(  [256,4096])
                legend = []
                for j in range(0,len(skeletal_ureter_sublist)):
                    current_meta    = process_x15_rt_file_name(skeletal_ureter_sublist[j])
                    current_content = process_file_content_x15_rt_two_step(skeletal_ureter_sublist[j])
                    if not current_content['incomplete_data']:
                        legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                        plt.plot(wl_set,current_content['x15_c'][:,skeletal_ureter_rois[j][2]])
                legend.append('baseline')
                plt.plot(wl_set,np.ones([n_wls]) * watch_baseline)
                for j in range(0,len(wl_set)):
                    plt.axvline(wl_set[j])
                plt.legend(legend)
                ure_save_at = os.path.join(raw_plots_folder , 'ure_centre_pixel.pdf')
                plt.savefig(ure_save_at)
                #plot raw signal levels.
                
                #plot air-reference normalized values.
                if reference_20_found and reference_30_found:
                    plt.figure()
                    plt.xlabel('wavelengths (nm)')
                    plt.ylabel('mean cmos count')
                    plt.title('Tx intensities - Ureters , normalized')
                    plt.ylim([0,1])
                    legend = []
                    for j in range(0,len(skeletal_ureter_sublist)):
                        current_meta       = process_x15_rt_file_name(skeletal_ureter_sublist[j])
                        current_content    = process_file_content_x15_rt_two_step(skeletal_ureter_sublist[j])
                        if not current_content['incomplete_data']:
                            current_normalized = np.divide(current_content['x15_c'],reference_read)
                            current_normalized = current_normalized/np.max(current_normalized)
                            legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                            plt.plot(wl_set,current_normalized[:,skeletal_ureter_rois[j][2]])
                    for j in range(0,len(wl_set)):
                        plt.axvline(wl_set[j])
                    plt.legend(legend)
                    ure_norm_save_at = os.path.join(air_ref_plots_folder,'ure_centre_pixel.pdf')
                    plt.savefig(ure_norm_save_at)
                #plot air reference normalized values.

                #plot slopes between neighbours.
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title( 'Slopes(2) - Ureters')
                legend = []
                for j in range(0,len(skeletal_ureter_sublist)):
                    current_meta    = process_x15_rt_file_name(skeletal_ureter_sublist[j])
                    current_content = process_file_content_x15_rt_two_step(skeletal_ureter_sublist[j])
                    if not current_content['incomplete_data']:
                        legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                        slopes_current = current_content['x15_c'][1:,skeletal_ureter_rois[j][2]] - current_content['x15_c'][:-1,skeletal_ureter_rois[j][2]]
                        plt.plot(wl_set[0:-1],slopes_current)
                for j in range(0,len(wl_set)-1):
                    plt.axvline(wl_set[j])
                plt.legend(legend)
                ure_save_at = os.path.join(raw_plots_folder , 'ure_slopes-2_centre_pixel.pdf')
                plt.savefig(ure_save_at)
                #plot slopes between neighbours.
            #Plot skeletal ureters.
            
            #Plot skeletal peritoneums.
            if len(peritoneum_sublist) > 0:
                
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title('Tx intensities - Peritoneums')
                plt.ylim([256,4096])
                legend = []
                for j in range(0,len(peritoneum_sublist)):
                    current_meta    = process_x15_rt_file_name(peritoneum_sublist[j])
                    current_content = process_file_content_x15_rt_two_step(peritoneum_sublist[j])
                    if not current_content['incomplete_data']:
                        legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                        plt.plot(wl_set,current_content['x15_c'][:,peritoneum_rois[j][2]])
                legend.append('baseline')
                plt.plot(wl_set,np.ones([n_wls]) * watch_baseline)
                for j in range(0,len(wl_set)):
                    plt.axvline(wl_set[j])
                plt.legend(legend)
                peri_save_at = os.path.join(raw_plots_folder , 'peri_centre_pixel.pdf')
                plt.savefig(peri_save_at)
                
                if reference_20_found and reference_30_found:
                    plt.figure()
                    plt.xlabel('wavelengths (nm)')
                    plt.ylabel('mean cmos count')
                    plt.title('Tx intensities - Peritoneums , normalized')
                    plt.ylim([0,1])
                    legend = []
                    for j in range(0,len(peritoneum_sublist)):
                        current_meta       = process_x15_rt_file_name(peritoneum_sublist[j])
                        current_content    = process_file_content_x15_rt_two_step(peritoneum_sublist[j])
                        if not current_content['incomplete_data']:
                            current_normalized = np.divide(current_content['x15_c'],reference_read)
                            current_normalized = current_normalized/np.max(current_normalized)
                            legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                            plt.plot(wl_set,current_normalized[:,peritoneum_rois[j][2]])
                    for j in range(0,len(wl_set)):
                        plt.axvline(wl_set[j])
                    plt.legend(legend)
                    peri_norm_save_at = os.path.join(air_ref_plots_folder,'peri_centre_pixel.pdf')
                    plt.savefig(peri_norm_save_at)

                #plot slopes between neighbours.
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title( 'Slopes(2) - Peritoneums')
                legend = []
                for j in range(0,len(peritoneum_sublist)):
                    current_meta    = process_x15_rt_file_name(peritoneum_sublist[j])
                    current_content = process_file_content_x15_rt_two_step(peritoneum_sublist[j])
                    if not current_content['incomplete_data']:
                        legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                        slopes_current = current_content['x15_c'][1:,peritoneum_rois[j][2]] - current_content['x15_c'][:-1,peritoneum_rois[j][2]]
                        plt.plot(wl_set[0:-1],slopes_current)
                for j in range(0,len(wl_set)-1):
                    plt.axvline(wl_set[j])
                plt.legend(legend)
                peri_save_at = os.path.join(raw_plots_folder , 'peri_slopes-2_centre_pixel.pdf')
                plt.savefig(peri_save_at)
                #plot slopes between neighbours.
            #Plot skeletal peritoneums.

            #Plot native ureters.
            if len(native_ureter_sublist) > 0:
                
                #plot raw signal levels.
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title( 'Tx intensities - Native Ureters')
                plt.ylim(  [256,4096])
                legend = []
                for j in range(0,len(native_ureter_sublist)):
                    current_meta    = process_x15_rt_file_name(native_ureter_sublist[j])
                    current_content = process_file_content_x15_rt_two_step(native_ureter_sublist[j])
                    if not current_content['incomplete_data']:
                        legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                        plt.plot(wl_set,current_content['x15_c'][:,native_ureter_rois[j][2]])
                legend.append('baseline')
                plt.plot(wl_set,np.ones([n_wls]) * watch_baseline)
                for j in range(0,len(wl_set)):
                    plt.axvline(wl_set[j])
                plt.legend(legend)
                ure_save_at = os.path.join(raw_plots_folder , 'native_ure_centre_pixel.pdf')
                plt.savefig(ure_save_at)
                #plot raw signal levels.
                
                #plot air-reference normalized values.
                if reference_20_found and reference_30_found:
                    plt.figure()
                    plt.xlabel('wavelengths (nm)')
                    plt.ylabel('mean cmos count')
                    plt.title('Tx intensities - Native Ureters , normalized')
                    plt.ylim([0,1])
                    legend = []
                    for j in range(0,len(native_ureter_sublist)):
                        current_meta       = process_x15_rt_file_name(native_ureter_sublist[j])
                        current_content    = process_file_content_x15_rt_two_step(native_ureter_sublist[j])
                        if not current_content['incomplete_data']:
                            current_normalized = np.divide(current_content['x15_c'],reference_read)
                            current_normalized = current_normalized/np.max(current_normalized)
                            legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                            plt.plot(wl_set,current_normalized[:,native_ureter_rois[j][2]])
                    for j in range(0,len(wl_set)):
                        plt.axvline(wl_set[j])
                    plt.legend(legend)
                    ure_norm_save_at = os.path.join(air_ref_plots_folder,'native_ure_centre_pixel.pdf')
                    plt.savefig(ure_norm_save_at)
                #plot air reference normalized values.

                #plot slopes between neighbours.
                plt.figure()
                plt.xlabel('wavelengths (nm)')
                plt.ylabel('mean cmos count')
                plt.title( 'Slopes(2) - Native Ureters')
                legend = []
                for j in range(0,len(native_ureter_sublist)):
                    current_meta    = process_x15_rt_file_name(native_ureter_sublist[j])
                    current_content = process_file_content_x15_rt_two_step(native_ureter_sublist[j])
                    if not current_content['incomplete_data']:
                        legend.append(current_meta['pig1'] + '-' + current_meta['spec1'] + '-' + current_meta['sample1'])
                        slopes_current = current_content['x15_c'][1:,native_ureter_rois[j][2]] - current_content['x15_c'][:-1,native_ureter_rois[j][2]]
                        plt.plot(wl_set[0:-1],slopes_current)
                for j in range(0,len(wl_set)-1):
                    plt.axvline(wl_set[j])
                plt.legend(legend)
                ure_save_at = os.path.join(raw_plots_folder , 'native_ure_slopes-2_centre_pixel.pdf')
                plt.savefig(ure_save_at)
                #plot slopes between neighbours.
            #Plot native ureters.