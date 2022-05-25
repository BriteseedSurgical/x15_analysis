import os
import csv
import glob
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt

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

#default seek indices
seek_leds_set_1 = [200,300,500,700,1000,1500,2000]
#default seek indices

#calib-adapt pixels
pixels_vis_calib_adapt = [4,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def divide(x):
    wls = []
    global n_wls
    global n_pixels

    for i in range(0,n_wls):
        wls.append(x[int(i*n_pixels):int((i+1)*n_pixels)])
    
    return np.array(wls)

def process_x15_rt_file_name(f , PRINT = False):

    f = f.split('/')[-1]

    meta                = {}
    keys                = ['vessel' , 'diameter' , 'tissue' , 'thickness' , 'baseline' , 'comment', \
                           'tool' , 'line' , 'user' , 'orientation' , 'approach' , 'ja_manual' ,   \
                           'pig1' , 'spec1', 'sample1' , 'pig2' , 'spec2' , 'sample2' , \
                           'position_1' , 'position_2' , 'clock']
    meta['keys']        = keys

    meta['vessel']      = f.split('_v_')[-1].split('_')[0]
    meta['diameter']    = f.split('_vs_')[-1].split('_')[0]
    meta['tissue']      = f.split('_t_')[-1].split('_')[0]
    meta['thickness']   = f.split('_th_')[-1].split('_')[0]
    meta['baseline']    = f.split('_b_')[-1].split('_')[0]
    meta['comment']     = f.split('_c_')[-1].split('_tool_')[0].split('_2step')[0]
    meta['tool']        = f.split('_tool_')[-1].split('_line_')[0]
    meta['line']        = f.split('_line_')[-1].split('_')[0]
    meta['user']        = f.split('_user_')[-1].split('_')[0]
    meta['orientation'] = f.split('_o_')[-1].split('_')[0]
    meta['approach']    = f.split('_app_')[-1].split('_')[0]
    meta['ja_manual']   = f.split('_ja_')[-1].split('_')[0]

    meta['pig1']        = f.split('_s1_')[-1].split('-')[0]
    if len(meta['pig1']) == 1:
        _ = 0
    else:
        meta['pig1'] = str(0)
    
    meta['spec1']       = f.split('_s1_' + meta['pig1'] + '-')[-1].split('-')[0]
    if len(meta['spec1']) == 1:
        _ = 0
    else:
        meta['spec1'] = str(0)
    
    meta['sample1']     = f.split('_s1_')[-1].split('_')[0].split('-')[-1]
    if len(meta['sample1']) == 1:
        _ = 0
    else:
        meta['sample1'] = str(0)

    meta['pig2']        = f.split('_s2_')[-1].split('-')[0]
    if len(meta['pig2']) == 1:
        _ = 0
    else:
        meta['pig2'] = str(0)
    
    meta['spec2']       = f.split('_s2_' + meta['pig2'] + '-')[-1].split('-')[0]
    if len(meta['spec2']) == 1:
        _ = 0
    else:
        meta['spec2'] = str(0)
    
    meta['sample2']     = f.split('_s2_')[-1].split('_')[0].split('-')[-1]
    if len(meta['sample2']) == 1:
        _ = 0
    else:
        meta['sample2'] = str(0)

    pix = f.split('_pix_')[-1].split('_')[0]

    if pix == '0' or pix == '1':
        pix_tgl = int(pix)
    else:
        pix_tgl = int(1)

    #pix_tgl = int(0)
    
    if pix_tgl == int(0):
        position1 = f.split('_pos1_')[-1].split('_')[0]
        position2 = f.split('_pos2_')[-1].split('_')[0]
    else:
        position1_start = str(int((int(f.split('_pos1_')[-1].split('-')[0]) * 98/20)+ 1))
        position1_end   = str(int((int(f.split('_pos1_')[-1].split('_')[0].split('-')[-1]) * 98/20)+ 1))
        position2_start = str(int((int(f.split('_pos2_')[-1].split('-')[0]) * 98/20)+ 1))
        position2_end   = str(int((int(f.split('_pos2_')[-1].split('_')[0].split('-')[-1]) * 98/20)+ 1))
        position1       = position1_start + '-' + position1_end
        position2       = position2_start + '-' + position2_end
        #position1 = f.split('_pos1_')[-1].split('_')[0]
        #position2 = f.split('_pos2_')[-1].split('_')[0]
    
    meta['position_1']  = position1
    meta['position_2']  = position2
    meta['clock']       = f.split('_clock_')[-1].split('.csv')[0]

    return meta

def process_file_content_x15_rt_two_step(f , plot_calib_bands = False , PRINT = False):
    
    #code : update_state = 0 -> NIR update
    #code : update_state = 1 -> Vis update

    #consistently as a first step , remove all duplicates.
    #Identified the problem  - Fewer than n_postconv rows saved to file.
    #Solution : 1. Remove duplicate rows. 2.Use converged flag alone. 3.If not converged , use last state as 
    #post conv state. 4. Pair up from the end.

    global n_wls , n_control , n_pixels , visible_end_index 
    global max_post_adapt_length #(should now be a variable 1 or 32 , or make decoding invariant to this.)

    processed = {}

    processed['incomplete_data'] = 1
    if PRINT:
        print('')

    datagram      = pd.read_csv(f).values
    n             = np.shape(datagram)[0]

    if n <= 2:
        processed['too_few_rows'] = 1
        if PRINT:
            print('Case of too few rows.')
        return processed
    #Do not process files with fewer than 2 rows. Should be made files that 
    #do have only a single flavour of LED states.

    processed['too_few_rows'] = 0
    processed['n_rows']       = n

    ls            = datagram[:,0:n_control]
    if PRINT:
        print('File last    : ' , ls[-1,:])
    cmos          = datagram[:,n_control:int(n_control + n_wls * n_pixels)]
    optim_count   = datagram[:,n_control + (n_wls * n_pixels) + 2 - 1]
    converged     = datagram[:,n_control + (n_wls * n_pixels) + 11 - 1]
    u_states      = datagram[:,-1] 

    if PRINT:
        print('Rows in file : ' , n)
        print('Converged    : ' , int(np.sum(converged)))

    n_reduced             = n
    post_conv_vis         = None
    post_conv_nir         = None
    final_post_conv_state = None

    if int(np.sum(converged)) == 0:
        
        processed['converged'] = 0

        j            = n-1
        ctr          = 0

        final_post_conv_state = ls[-1,:]
        post_conv_vis         = vis_read(final_post_conv_state)
        post_conv_nir         = nir_read(final_post_conv_state)
        if PRINT:
            print('Postconv l     : ' , final_post_conv_state)
            print('Postconv vis ? : ' , post_conv_vis)
            print('Postconv nir ? : ' , post_conv_nir)
        
        post_conv_cmos_reads  = []
        while j > -1:

            if ctr < max_post_adapt_length and np.array_equal(ls[j] , final_post_conv_state):
                post_conv_cmos_reads.append(cmos[j,:])
                j   = j-1
                ctr = ctr + 1
            else:
                break
        
        n_reduced             = int(n - ctr + 1) #origin of error // March 7 , 2022.
        
        if PRINT:
            print('postconv ctr : ' , ctr)
            print('Original     : ' , ls[n_reduced-2])
            print('Now          : ' , ls[n_reduced-1])
            print('n_red 0      : ' , n_reduced)

        mean_post_conv_read   = divide(np.array(np.mean(np.stack(post_conv_cmos_reads) , axis = 0)))

        processed['post_conv_state'] = final_post_conv_state
        processed['post_conv_read']  = mean_post_conv_read

    else:
        processed['converged']       = 1
        processed['post_conv_state'] = -1
        processed['post_conv_read']  = -1


    unique_leds           = []
    unique_cmos           = []
    reduced_update_states = []
    for j in range(0,n_reduced):
        
        if j == 0:
            unique_leds.append(ls[0,:])
            unique_cmos.append(cmos[0,:])
            reduced_update_states.append(int(u_states[0]))
        else:
            if np.array_equal(ls[j,:] , unique_leds[-1]):
                _ = 0
            else:
                unique_leds.append(ls[j,:])
                unique_cmos.append(cmos[j,:])
                reduced_update_states.append(int(u_states[j])) 

    n_reduced = len(unique_leds) #this to record
    if PRINT:
        print('n red 2      : ' , n_reduced)

    if PRINT:
        print('final now    : ' , unique_leds[-1])

    unique_nir_leds = []
    unique_nir_cmos = []

    unique_vis_leds = []
    unique_vis_cmos = []

    toggler         = 0 # true is the last element
    neither_ctr     = 0

    for j in range(0,len(unique_leds)):
        k = len(unique_leds) - j - 1
        if vis_read(unique_leds[k]):
            unique_vis_leds.append(unique_leds[k])
            unique_vis_cmos.append(divide(unique_cmos[k]))
            toggler = toggler + k
        elif nir_read(unique_leds[k]):
            unique_nir_leds.append(unique_leds[k])
            unique_nir_cmos.append(divide(unique_cmos[k]))
            toggler = toggler - k
        else:
            neither_ctr = neither_ctr + 1
    
    processed['neither_ctr'] = neither_ctr
    processed['toggler']     = toggler

    if PRINT:
        print('Vis length   : ' , len(unique_vis_leds))
        print('NIR length   : ' , len(unique_nir_leds))

    residue = -1
    n_vis = len(unique_vis_leds)
    n_nir = len(unique_nir_leds)
    if n_vis > n_nir:
        if n_vis == n_nir + 1:
            unique_vis_leds.pop()
            unique_vis_cmos.pop()
        else:
            residue = n_vis - n_nir
            if PRINT:
                print('Unexpected encounter Type 1.')
            for j in range(0,residue):
                unique_vis_leds.pop()
                unique_vis_cmos.pop()
    elif n_nir > n_vis:
        if n_nir == n_vis + 1:
            unique_nir_leds.pop()
            unique_nir_cmos.pop()
        else:
            residue = n_nir - n_vis
            if PRINT:
                print('Unexpected encounter Type 2.')
            for j in range(0,residue):
                unique_nir_leds.pop()
                unique_nir_cmos.pop()
    else:
        _ = 0
    
    if PRINT:
        print('')
        print('Vis length   : ' , len(unique_vis_leds))
        print('NIR length   : ' , len(unique_nir_leds))

    if len(unique_vis_leds) > 0:
        if PRINT:
            print('FvisL        : ' , unique_vis_leds[0])
    if len(unique_nir_leds) > 0:
        if PRINT:
            print('FnirL        : ' , unique_nir_leds[0])

    processed['residue']          = residue
    processed['final_vis_length'] = len(unique_vis_leds)
    processed['final_nir_length'] = len(unique_nir_leds)
    processed['unique_vis_leds']  = np.array(unique_vis_leds)
    processed['unique_vis_cmos']  = np.array(unique_vis_cmos)
    processed['unique_nir_leds']  = np.array(unique_nir_leds)
    processed['unique_nir_cmos']  = np.array(unique_nir_cmos)

    if len(unique_vis_leds) < 1 or len(unique_nir_leds) < 1:
        processed['incomplete_data'] = 1
        return processed

    processed['incomplete_data'] = 0

    x15_leds = []
    x15_cmos = []
    for j in range(0,processed['final_vis_length']):
        L = [int(processed['unique_vis_leds'][j,0]),int(processed['unique_nir_leds'][j,1]),int(processed['unique_vis_leds'][j,2])\
            ,int(processed['unique_nir_leds'][j,3]),int(processed['unique_vis_leds'][j,4]),int(processed['unique_nir_leds'][j,5])]
        C = np.zeros([n_wls , n_pixels])
        C[0:visible_end_index,:]     = processed['unique_vis_cmos'][j,0:visible_end_index,:]
        C[visible_end_index:n_wls,:] = processed['unique_nir_cmos'][j,visible_end_index:n_wls,:]
        x15_leds.append(L)
        x15_cmos.append(C)

    x15_leds = np.array(x15_leds)
    x15_cmos = np.array(x15_cmos)

    processed['x15_ls'] = x15_leds
    processed['x15_cs'] = x15_cmos
    
    if PRINT:
        print('Shapes final : ' , np.shape(x15_leds) , np.shape(x15_cmos))

    processed['l']     = x15_leds[0,:]
    processed['x15_c'] = x15_cmos[0,:,:]
  
    if PRINT:
        print('Shapes(l,c)  : ' , np.shape(processed['l']) , np.shape(processed['x15_c']))

    if plot_calib_bands:
        plt.figure()
        plt.ylim([400,4096])
        plt.title(x15_leds[0,:])
        plt.plot(x15_cmos[0,4,:],'g-')
        plt.plot(x15_cmos[0,14,:],'r-')
        plt.plot(np.ones([99])*1600,'b-')
        plt.show()

    if PRINT:
        print('')

    return processed

def analyze_signal_levels(c , b = 1600 , index_vis = 4 , index_nir = n_wls - 1):
    #input : processed (aggregated) raw (un normalized) cmos array
    
    analysis = {}

    # Adapt scores consider only adapted bands.
    # define a dark count (we're looking for 2-3x of this as a mean intensity of each band at the end of an adapt).
    
    # a measure of speed ? and convergence : speed and convergence - get outside in main loop.
    # record n_unique_cycles.
    # record the 6 distances.
    # record mean deviation of vis ref and mean deviation of nir ref. 
    # a new processing function taking in the entire adapt sequence.
    # Mean vis error vs time , Mean nir error vs time 
    # Compute the change in error and see when it falls below a threshold.
    # Make the adapt convergence plots.
    global n_wls
    
    band_means = []
    band_stds  = []
    band_mins  = []
    band_maxs  = []

    for j in range(0,n_wls):
        band_means.append(int(np.mean(c[j,:])))
        band_stds.append(int(np.std(c[j,:])))
        band_mins.append(int(np.min(c[j,:])))
        band_maxs.append(int(np.max(c[j,:])))

    analysis['mins'] = band_mins
    analysis['maxs'] = band_maxs
    analysis['means']= band_means
    analysis['stds'] = band_stds

    mean_vis  = np.mean(c[index_vis,:])
    mean_nir  = np.mean(c[index_nir,:])

    vis_shift = mean_vis - b
    nir_shift = mean_nir - b 

    vis_dev_l1= np.mean(np.abs(c[index_vis,:] - b))
    nir_dev_l1= np.mean(np.abs(c[index_nir,:] - b))

    analysis['vis_shift']  = int(vis_shift)
    analysis['nir_shift']  = int(nir_shift)
    analysis['vis_dev_l1'] = int(vis_dev_l1)
    analysis['nir_dev_l1'] = int(nir_dev_l1)

    return analysis

def make_rt_mat(rt_dict , meta , fp = [] , generate_fp = False):
    
    rt_mat = {}

    if generate_fp:
        for key in rt_dict:
            rt_mat[key] = rt_dict[key]
        rt_mat['meta']        = meta
        rt_mat['x15_fp']      = fp
        rt_mat['x15_fp_norm'] = np.divide(rt_dict['x15_c'],fp)
    else:
        for key in rt_dict:
            rt_mat[key] = rt_dict[key]
        rt_mat['meta']        = meta
    return rt_mat

def generate_metrics_group_1(l , c):
    #On raw data -> Hari's metrics and my FP metrics invalid.

    metric_set = []
    n_pixels = np.shape(c)[1]
    
    M  = [l[0]/l[1] , l[2]/l[3] , l[4]/l[5]]
    
    s1 = (1/n_pixels) * (np.sum(np.abs(c[x4_band_indices[3],:] - c[x4_band_indices[2],:]))) 
    s2 = (1/n_pixels) * (np.sum(c[x4_band_indices[3],:] - c[x4_band_indices[2],:]))
    S1 = s2/s1

    s3 = (1/n_pixels) * (np.sum(np.abs(c[x4_band_indices[3],:] - c[x4_band_indices[1],:]))) 
    s4 = (1/n_pixels) * (np.sum(c[x4_band_indices[3],:] - c[x4_band_indices[1],:]))
    S2 = s4/s3

    S3 = np.median(np.divide(c[6,:],c[3,:]))
    S4 = np.median(np.divide(c[7,:],c[4,:]))
    S5 = np.median(np.divide(c[n_wls-1,:],c[4,:]))

    metric_set = [round(S1,3) , round(S2,3) , round(S3,3) , round(S4,3) , round(S5,3),round(M[0],3) , round(M[1],3) , round(M[2],3)]

    return metric_set

#obtain calibration data at discrete LED Intensities , from all cases.
def read_calib_data_discrete_leds(calib_folder , target_led_list):
    
    analysis = {}

    global n_wls , n_control , n_pixels , visible_end_index , max_post_adapt_length

    targets   = glob.glob(os.path.join(calib_folder,'*.csv'))
    n_targets = len(targets)

    if n_targets == 0:
        print('Empty target folder.')
    else:
        if n_targets == 9:
            print('9 files found. Expected from a normal calib folder.')
        else:
            print('Not the usual full calibration set.')

    for j in range(0,n_targets):
        case = targets[j].split('/')[-1].split('_led_')[-1].split('.csv')[0]
        rep_led = int(case[0]) - 1

        datagram      = pd.read_csv(targets[j]).values
        n             = np.shape(datagram)[0]

        leds = datagram[:,0:n_control]
        cmos = datagram[:,n_control:n_control + (n_wls * n_pixels)]

        d = {}
        for k in range(0,n):
            for l in range(0,len(target_led_list)):
                if np.array_equal(leds[k,rep_led],target_led_list[l]):
                    d[str(target_led_list[l])] = divide(cmos[k,:])
                    _ = 0
                    #correct for artifact ? Lipshitzness as a measure ?

        analysis[case] = d
    print('')
    return analysis
#obtain calibration data at discrete LED Intensities , from all cases.

#obtain calibration data over the LED sweep , of discrete bands and at 
#discrete pixels across the sensor.
def read_calib_data(calib_folder , vis_index = 4, interest_vis = pixels_vis_calib_adapt, nir_index = n_wls - 1 , interest_nir = pixels_nir_calib_adapt):
    #vis and nir indices are integers , pixels of interest are lists of variable lengths
    #currently there are no guards against entering in an invalid index or PoI.

    relevant = {}

    global n_wls , n_control , n_pixels , visible_end_index , max_post_adapt_length

    targets   = glob.glob(os.path.join(calib_folder,'*.csv'))
    n_targets = len(targets)

    if n_targets == 0:
        print('Empty target folder.')
    else:
        if n_targets == 9:
            print('9 files found. Expected from a normal calib folder.')
        else:
            print('Not the usual full calibration set.')

    for j in range(0,n_targets):
        
        case     = targets[j].split('/')[-1].split('_led_')[-1].split('.csv')[0]
        print('Processing case of LED(s) : ' , case)
        rep_led  = int(case[0]) - 1

        datagram = pd.read_csv(targets[j]).values
        n        = np.shape(datagram)[0]

        leds     = datagram[:,0:n_control]
        cmos     = datagram[:,n_control:n_control + (n_wls * n_pixels)]

        #make it standard practice to put keys into output dictionary.
        current      = {}
        current['l'] = leds[:,rep_led]

        #for the vis band  : do it for all cases ? - I think yes
        #can filter in the main loop or have it as a floor read.
        for k in range(0,len(interest_vis)):
            pixel = interest_vis[k]
            title = 'p_' + str(pixel) + '_b_' + str(wl_set[vis_index])
            band  = np.zeros([n])
            for l in range(0,n):
                band[l] = (divide(cmos[l,:]))[vis_index,interest_vis[k]]
            current[title]  = band 

        #for the nir band  : do it for all cases ? - I think yes
        #can filter in the main loop or have it as a floor read.
        for k in range(0,len(interest_nir)):
            pixel = interest_nir[k]
            title = 'p_' + str(pixel) + '_b_' + str(wl_set[nir_index])
            band  = np.zeros([n])
            for l in range(0,n):
                band[l] = (divide(cmos[l,:]))[nir_index,interest_nir[k]]
            current[title]  = band 

        relevant[case] = current            
            
    print('')

    return relevant
#obtain calibration data over the LED sweep , of discrete bands and at 
#discrete pixels across the sensor.

def plot_x15_raw(c , save_into , baseline , title = 'Spatial profile',SHOW = False):
    
    global n_wls , n_pixels 

    if np.shape(c)[0] == n_wls and np.shape(c)[1] == n_pixels:
        for j in range(0,n_wls):
            plt.figure()
            plt.title(title)
            plt.xlabel('pixel count')
            plt.ylabel('cmos count')
            plt.ylim([256,4096])
            plt.plot(c[j,:])
            plt.plot(np.ones([n_pixels]) * baseline)
            plt.legend([str(wl_set[j]) + ' nm' , 'baseline'])
            plt.savefig(os.path.join(save_into, str(wl_set[j]) + '.png'))
            if SHOW:
                plt.show()
    else:
        print('')
        print('Unexpected input encountered.')
        print('')
    
    return

def plot_x15_spatial(c ,baseline , title = 'Spatial profile' , save_to = 'x15_spatial_def.pdf' , save = False , show = True):
    
    global n_wls , wl_set , n_pixels

    plt.figure()
    plt.xlabel('pixel count')
    plt.ylabel('cmos intensity')
    plt.xlim([0,99])
    plt.ylim([256,4096])
    plt.title(title)
    for j in range(0,n_wls):
        plt.plot(c[j,:])
    plt.legend(wl_set)
    if save:
        plt.savefig(save_to)
    if show:
        plt.show()

    return

def analyze_adapt(cmos_sequence , b = 1600 , index_vis = 4 , index_nir = n_wls - 1):
    
    #input : processed (aggregated) raw (un normalized) sequence of cmos arrays
    
    analysis = {}
    # Adapt scores consider only adapted bands.
    # define a dark count (we're looking for 2-3x of this as a mean intensity of each band at the end of an adapt).
    # record n_unique_cycles.
    # record the 6 distances.
    # record mean deviation of vis ref and mean deviation of nir ref. 
    # a new processing function taking in the entire adapt sequence.
    # Mean vis error vs time , Mean nir error vs time. 
    # Make and store these plots in memory.
    # Compute the change in error and see when it falls below a threshold.
    # Make the adapt convergence plots.

    global n_wls , n_pixels , pixels_vis_calib_adapt , pixels_nir_calib_adapt
    c             = cmos_sequence[0,:,:]
    n_cycles      = np.shape(cmos_sequence)[0]
    
    d_vis_end     = [int(c[index_vis,pixels_vis_calib_adapt[j]] - b) for j in range(0,len(pixels_vis_calib_adapt))]
    d_nir_end     = [int(c[index_nir,pixels_nir_calib_adapt[j]] - b) for j in range(0,len(pixels_nir_calib_adapt))]

    l1_vis_end    = int(np.mean(np.abs(d_vis_end))) #L1 norm of error at runtime/discrete pixels
    l1_nir_end    = int(np.mean(np.abs(d_nir_end))) #L1 norm of error at runtime/discrete pixels

    #Convergence yes or no , in main loop.

    d_vis         = [[np.abs(cmos_sequence[j,index_vis,pixels_vis_calib_adapt[k]] - b) for j in range(0,n_cycles)] for k in range(0,len(pixels_vis_calib_adapt))]
    d_nir         = [[np.abs(cmos_sequence[j,index_nir,pixels_nir_calib_adapt[k]] - b) for j in range(0,n_cycles)] for k in range(0,len(pixels_nir_calib_adapt))]


    l1_vis_errors = [0.33 * (d_vis[0][k] + d_vis[1][k] + d_vis[2][k]) for k in range(0,n_cycles)]            
    l1_nir_errors = [0.33 * (d_nir[0][k] + d_nir[1][k] + d_nir[2][k]) for k in range(0,n_cycles)]
    print('Shapes : ' , np.shape(l1_vis_errors) , np.shape(l1_nir_errors))
    #Plots inside or out ? 

    mean_vis      = np.mean(c[index_vis,:])
    mean_nir      = np.mean(c[index_nir,:])

    vis_shift     = mean_vis - b #mean signal level - baseline
    nir_shift     = mean_nir - b #mean signal level - baseline

    vis_dev_l1    = np.mean(np.abs(c[index_vis,:] - b)) #overall L1 norm of error (Vis band)
    nir_dev_l1    = np.mean(np.abs(c[index_nir,:] - b)) #overall L1 norm of error (NIR band)

    #put everything into analysis.
    #plot the two plots outside this loop.
    #Run the analysis for the primary (1600) baseline.

    #can get calibration shifts of the bands. [right shifts - > shifts of sensitivity]

    analysis['n_cycles']      = int(n_cycles)
    analysis['d_vis_end']     = d_vis_end
    analysis['d_nir_end']     = d_nir_end
    analysis['err_rt_vis']    = int(l1_vis_end)
    analysis['err_rt_nir']    = int(l1_nir_end)

    analysis['mean_vis']      = int(mean_vis)
    analysis['mean_nir']      = int(mean_nir)
    
    analysis['vis_shift']     = int(vis_shift)
    analysis['nir_shift']     = int(nir_shift)
    
    analysis['vis_dev_l1']    = int(vis_dev_l1)
    analysis['nir_dev_l1']    = int(nir_dev_l1)

    #the 6 distances. Do it tomorrow first thing.

    analysis['l1_vis_errors'] = l1_vis_errors
    analysis['l1_nir_errors'] = l1_nir_errors

    return analysis

def plot_x15_adapt(vis,nir,baseline,title = 'Reference bands at adapt-end',pix_vis = pixels_vis_calib_adapt,\
                   pix_nir = pixels_nir_calib_adapt,save_at = 'adapt_placeholder.png', \
                   save = True , show = False):
    plt.figure()
    plt.xlabel('pixel count')
    plt.ylabel('cmos count')
    plt.xlim([1,99])
    plt.ylim([256,4096])
    plt.title(title)
    plt.plot(vis,'b-')
    plt.plot(nir,'r-')
    plt.plot(np.ones([n_pixels]) * baseline,'g-')
    for j in range(0,len(pix_vis)):
        plt.axvline(pix_vis[j],color='b')
    for j in range(0,len(pix_nir)):
        plt.axvline(pix_nir[j],color='r')
    plt.legend(['vis reference','nir reference','baseline'])
    if save:
        plt.savefig(save_at)
    if show:
        plt.show()
    return

def verify_residual_slope(ip_sequence , start , threshold):
    n = len(ip_sequence)
    check = True
    for j in range(start , n):
        if np.abs(ip_sequence[j]) > threshold:
            check = False
            break
    return check

def get_mean_levels(c , start = 0 , end = n_pixels):
    
    # Track mean levels over time ? 
    mean_levels = None
    n = np.shape(c)
    if len(n) == 2:
        mean_levels = np.mean(c[:,start:end],axis = 1)
    else:
        mean_levels = []

    return mean_levels

def parse_spectrometer_file(f , header_length = 12, title = 'spectrometer_response', \
                            save_at = 'spectrometer_default.png' , vert = True , \
                            plot_discrete = True,plot = False , save = False):
    wavelengths    = []
    tx_intensities = []

    datagram = pd.read_csv(f)
    datagram = datagram.to_csv(f.split('.txt')[0] + '.csv',index = None)
    
    datagram = pd.read_csv(f.split('.txt')[0] + '.csv').values
    datagram = datagram[header_length:]

    n = len(datagram)
    for j in range(0,n):
        wavelengths.append(datagram[j][0].split('\t')[0])
        tx_intensities.append(datagram[j][0].split('\t')[-1])

    for j in range(0,n):
        wavelengths[j]    = float(wavelengths[j])
        tx_intensities[j] = float(tx_intensities[j])

    global n_wls , wl_set
   
    tx_intensities_reduced = []
    #ideally we should be integrating gaussians.

    for j in range(0,len(wl_set)):

        for k in range(0,len(wavelengths)):
            if int(wavelengths[k]) == wl_set[j]:
                tx_intensities_reduced.append(tx_intensities[k])
                break
    tx_intensities_reduced.append(tx_intensities[-10])

    if plot_discrete:
        plt.figure()
        plt.xlabel('wavelength (nm)')
        plt.ylabel('Rx intensity (absolute)')
        plt.ylim([0,60000])
        plt.title(title)
        plt.plot(wl_set,tx_intensities_reduced,color='g')
        if vert:
            for i in range(0,n_wls):
                plt.axvline(wl_set[i])
        plt.xticks(wl_set , rotation = 45)
        if save:
            plt.savefig(save_at)
        if plot:
            plt.show()    
    else:
        #draw the vertical lines of our 15 WLs.
        plt.figure()
        plt.xlabel('wavelength (nm)')
        plt.ylabel('Rx intensity (absolute)')
        plt.title(title)
        plt.plot(wavelengths,tx_intensities,color='g')
        if vert:
            for i in range(0,n_wls):
                plt.axvline(wl_set[i])
        #plt.xticks(wl_set , rotation = 45)
        if save:
            plt.savefig(save_at)
        if plot:
            plt.show()
    

    return [wavelengths , tx_intensities]

def plot_mean_levels(cmos,start = 0,stop = n_pixels,\
                     baseline = 1600,title = 'mean_signal_levels',\
                     save_at = 'default_mean_levels_plot.pdf',\
                     vert = True,save = True , show = False):

    global n_wls
    mean_levels_entire = np.mean(cmos,axis = 1)
    mean_levels_in_roi = np.mean(cmos[:,start:stop],axis = 1)
    plt.figure()
    plt.title(title)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Mean Rx intensity')
    plt.ylim([256,4096])
    plt.plot(wl_set,mean_levels_entire)
    plt.plot(wl_set,mean_levels_in_roi)
    plt.plot(wl_set,np.ones([n_wls]) * baseline)
    plt.xticks(wl_set,rotation = 45)
    if vert:
        for j in range(0,len(wl_set)):
            plt.axvline(wl_set[j])
    plt.legend(['over all pixels' , 'in primary RoI' , 'baseline'])
    if save:
        plt.savefig(save_at)
    if show:
        plt.show()

    return

if __name__ == '__main__' : 

    print('')
    root        = os.getcwd()
    parent      = Path(root)
    parent      = parent.parent.absolute()
    data_path   = os.path.join(parent,'docs')
    data_path   = os.path.join(data_path , 'oo_tissue_reads')

    plots_path    = os.path.join(parent,'x15_rt_processed')
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    plots_path    = os.path.join(plots_path,'spectrometer_plots_x15')
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    experiments = glob.glob(os.path.join(data_path,'*'))

    for i in range(0,len(experiments)):
        current_experiment = experiments[i]

        experiment_name      = current_experiment.split('/')[-1]
        current_plots_folder = os.path.join(plots_path,experiment_name)
        if not os.path.exists(current_plots_folder):
            os.mkdir(current_plots_folder)

        current_plots_folder_discrete = os.path.join(plots_path,experiment_name + '_discrete')
        if not os.path.exists(current_plots_folder_discrete):
            os.mkdir(current_plots_folder_discrete)

        samples = glob.glob(os.path.join(current_experiment,'*.txt'))
        print('')
        print('Current experiment : ' , current_experiment)
        print('File count         : ' , len(samples))
        
        for j in range(0 , len(samples)):
            sample_name = samples[j].split('/')[-1].split('.txt')[0]
            print('Sample name : ' , current_experiment + '-' + sample_name)
            _ = parse_spectrometer_file(samples[j] , header_length = 12 ,\
                                        title = 'spectrometer_response_discrete_' + sample_name,  \
                                        save_at = os.path.join(current_plots_folder_discrete,sample_name + \
                                        '.pdf') ,vert = False, plot_discrete = True,\
                                        plot = False , save = True)

            _ = parse_spectrometer_file(samples[j] , header_length = 12 ,\
                                        title = 'spectrometer_response_' + sample_name,  \
                                        save_at = os.path.join(current_plots_folder,sample_name + \
                                        '.pdf') ,vert = True, plot_discrete = False,\
                                        plot = False , save = True)