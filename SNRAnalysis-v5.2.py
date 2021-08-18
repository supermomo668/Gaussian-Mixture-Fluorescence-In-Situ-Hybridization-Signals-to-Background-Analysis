#!/usr/bin/env python
# coding: utf-8

# In[48]:

import os,  numpy as np, random, skimage.io, matplotlib.pyplot as plt
from skimage.filters import rank,  threshold_local, threshold_otsu
from skimage.morphology import disk
from skimage.feature import peak_local_max
import numpy as np, scipy.ndimage as ndi , PIL.Image as pil, cv2, pandas as pd
import argparse, matplotlib
from pathlib import Path 
from skimage.morphology import area_closing, closing, diameter_closing, remove_small_objects
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from skimage.color import label2rgb
import random, itertools as it
import configparser
import scipy 

matplotlib.use('TkAgg')

WORKING_DIR = os.getcwd()

def list_img_path(parent_path):
    path_imgs = []
    for file in os.listdir(parent_path):
        if file.endswith(".tif"):
            #print("Is image")
            path_imgs.append(parent_path/file)
    return path_imgs

def plot_gaussian(mean, std, im, res=100, user_thres=5000):
    x = im.ravel()
    xmin, xmax = np.min(x), np.max(x)
    xmax = min(user_thres, xmax)
    x = np.linspace(xmin, xmax, res)
    y = norm.pdf(x, mean, std)
    return x, y

def cross_image(p1: Path, p2: Path):
    """
    return (y_shift, x_shift) of fn1 relative to fn2
    """
    try:
        im1 = cv2.imread(str(p1), -1).astype('float')
        im2 = cv2.imread(str(p2), -1).astype('float')
    except:
        raise Exception('[Warning]Issues with loading the image,\
                        check background path:\n',p1)
    im_shape = im1.shape
    # get rid of the averages, otherwise the results are not good
    im1 -= np.mean(im1); im2 -= np.mean(im2)
    # calculate the correlation image; note the flipping of onw of the images
    corr_img = scipy.signal.fftconvolve(im1, im2[::-1,::-1], mode='same')
    max_corr_loc = np.unravel_index(np.argmax(corr_img), corr_img.shape)
    return max_corr_loc - np.array(list(map(lambda x: x/2, (im_shape))))

def find_shift(summary_df, cyc_num=1):
    # Picking FOVs (must do a few to ensure)
    reg_rows  = summary_df.loc[[i for i in summary_df.index if 
                            (i.endswith("L473S") and (str(cyc_num)+"R") in i)]]
    num_of_registers = max(1, min(10, int(len(reg_rows)*0.1)))
    print("[PROCESS]Registering with ", num_of_registers, "FOVs")
    if len(reg_rows) == 0:
        raise Exception("[Warning]You need L473 from Cyc01R images to register. Otherwise, set Use_bg=0")
    reg_shifts = []
    for i in range(max(1, num_of_registers)):  # pick 5 fov
        try: reg_row = reg_rows.iloc[i]
        except:  reg_row = summary_df.iloc[i]
        reg_shifts.append(cross_image(reg_row["Bg-Path"], reg_row["Path"]))
    x_shs = [reg_shifts[i][0] for i in range(len(reg_shifts))]
    y_shs = [reg_shifts[i][1] for i in range(len(reg_shifts))]
    x_sh = int(np.median(x_shs)); y_sh = int(np.median(y_shs))
    # Do image shifting and saving
    if (int(x_sh) | int(y_sh)) == 0:
        print("[PROCESS]No shift detected.")
    else:
        print("[PROCESS]Using shift", x_sh, y_sh)
    return x_sh , y_sh

def subtract_bg(x_sh:int, y_sh:int, cyc_img_path:Path, bg_img_path:Path):
    print("[Process]Background subtraction")
    '''
    Returns
    Added cyc image shift
    '''
    def shift2coord(shifts): # bg_img
        # shift coordinates for image based on shift
        xy_shift = []
        for sh in shifts:  # Do for x and y
            st_sh, end_sh = 0, 0; 
            if sh < 0: st_sh = abs(sh)
            elif sh > 0: end_sh = -sh
            xy_shift.append((st_sh, end_sh))
        return xy_shift
    bg_x_sh, bg_y_sh = shift2coord((x_sh, y_sh)); cyc_x_sh, cyc_y_sh = shift2coord((-x_sh, -y_sh))
    cyc_img = cv2.imread(str(cyc_img_path),-1); bg_img = cv2.imread(str(bg_img_path),-1)
    s = cyc_img.shape[0]
    shifted_cyc_img = cyc_img[cyc_x_sh[0]:s+cyc_x_sh[1], cyc_y_sh[0]:s+cyc_y_sh[1]]
    shifted_bg_img = bg_img[cyc_x_sh[0]:s+cyc_x_sh[1], cyc_y_sh[0]:s+cyc_y_sh[1]]
    print("[INFO] BG Image stat:", shifted_bg_img.mean(), shifted_bg_img.min(), shifted_bg_img.max())
    print("[INFO] Cyc Image stat:", shifted_cyc_img.mean(), shifted_cyc_img.min(), shifted_cyc_img.max())
    bg_subtracted_img = cv2.subtract(shifted_cyc_img, shifted_bg_img)
    bg_subtracted_img[np.where(bg_subtracted_img<0)] = 0
    return bg_subtracted_img,  (cyc_x_sh, cyc_y_sh)

def collect_channel_pixel(summary_df, ch="L532S"):
    channel_pixels = []
    ch_df = summary_df.loc[summary_df["Channel"]==ch]
    for ch_path in ch_df["Path"]:
        channel_pixels.append(cv2.imread(str(ch_path), -1).ravel())
    return np.array(channel_pixels)

def gmm(im, k=2):
    '''
    Gaussian Mixture Fitting onto pixel intensities.
    Rule out background subtracted image should intensities be below 0.
    return sorted labels based on intesnity
    '''
    gm = GaussianMixture(n_components=k, random_state=0)
    pred_labels = gm.fit_predict(im.ravel().reshape(-1,1))
    label_img = pred_labels.reshape(im.shape)
    # rank labels from low to high intesnity
    label_means = []; labels = np.arange(k)
    print("[INFO]Guassian Class:")
    for l in labels:
        print("No. Label",l,":", len(im[np.where(label_img==l)]))
        label_means.append(im[np.where(label_img==l)].mean())
    ind = np.argsort(label_means); sorted_labels = labels[ind] #; labels[zero_ind] = 0 # Labels are already 0
    return label_img, gm.means_.ravel()[ind], gm.covariances_.ravel()[ind], sorted_labels

def solve_gaussian(m, std):
    '''Solve Gaussian intersect
    m: list of means, std: list of stds
    '''
    assert len(m)==len(std)==2
    try:  # Find intersect
        a = 1/(2*std[0]**2) - 1/(2*std[1]**2)
        b = m[1]/(std[1]**2) - m[0]/(std[0]**2)
        c = m[0]**2 /(2*std[0]**2) - m[1]**2 / (2*std[1]**2) - np.log(std[1]/std[0])
        return np.roots([a,b,c])
        
    except np.linalg.LinAlgError: 
        print('[Warning]Failed to find roots for coefficients:',
              (a,b,c),".\nDefaulting to mid-point.")
        return np.array([m.sum()/len(m)])
    

class SNRAnalysis:
    def __init__(self, exp_path, settings):        
        self.exp_dir = exp_path
        ## User settings
        # Use either FOVs over Random sampling
        if settings.get('FOV', False) and len(settings.get('FOV', False).split(','))>0:
            self.FOVs = settings.get('FOV', False).split(','); self.num_ranSamples = False
        else:
            self.FOVs = False
            self.num_ranSamples = int(settings.get('NumberSamples', 30))
        # Montage
        self.use_montage = int(settings.get('Use_montage', True))
        
        # Background subtraction
        bg_img_path =exp_path/settings.get('bg_folder','bg')/'AcqData'
        self.use_bg = int(settings.get('Use_bg', False))
        if self.use_bg and os.path.exists(bg_img_path): 
            self.bg_img_path = bg_img_path; self.use_bg = True
            print("[INFO]Found existing background folder:",bg_img_path)
        else: 
            self.bg_img_path=None; self.use_bg = False
            print("[INFO] Not using background subtraction method.")
        #
        self.output_bgsub = bool(settings.get('save_bgsub', False))
        self.bgsub_fn = "BG-subtracted"
        # If directory is in AcqData
        if os.path.exists(self.exp_dir/"AcqData"):
            self.single_run = True; print("[INFO]Performing single folder run on 'AcqData' folder level.")
            self.use_bg=self.FOVs=False; self.num_ranSamples = int(settings.get('NumberSamples', 30))
            self.bg_img_path=None
        else: self.single_run=False
        #
        self.settings = settings; print("[Settings]", self.settings.items())
        
    def quick_analyze2_by_channel(self, img_paths,
                                  avail_chs=["L473S","L532S","L595S","L647S"], plot=False):
        summary_df = pd.DataFrame(columns=['Img','Path','Channel','Bg-Path', 'Mean', 'Std',
                'Fg mean', 'Bg mean', 'Fg-Bg SNR', 'Spot mean', 'Spot-fg SNR', 'Spot-bg SNR',
                '(M)Spot mean', '(M)Spot-fg SNR', '(M)Spot-bg SNR',
                '(BG-sub)Spot mean','(BG-sub)Bg mean','(BG-sub)SBR','Shift', 'Comment'])
        fp = True
        for img_path in img_paths:
            fov_name = img_path.stem
            if "DAPI" in fov_name:
                continue
            # background path
            if self.use_bg:
                if os.path.exists(self.bg_img_path/img_path.name):
                    bg_img_path = self.bg_img_path/img_path.name
                    if fp: print('[INFO]Background image found:'); fp=False
                    print(img_path.name)
                else: 
                    bg_img_path = None; print('[Warning]No background image:',img_path.name)
            else: bg_img_path = None
            img = cv2.imread(str(img_path), -1)
            mean, std = norm.fit(img.ravel())
            ch = [ch for ch in avail_chs if ch in fov_name][0]
            # Add to dataframe
            summary_df = summary_df.append({"Img": img_path.parent.parent.name+"_"+fov_name, "Path":img_path,
                                           'Channel':ch, "Bg-Path": bg_img_path, "Mean":mean,"Std":std},
                                           ignore_index=True)
        summary_df.set_index("Img", inplace=True)
        if plot:
            fig, ax= plt.subplots()
            for ch in avail_chs:
                all_ch_pixels = collect_channel_pixel(summary_df, ch)
                print("Shape of all pixel from the channel", all_ch_pixels.shape)
                mean, std = norm.fit(all_ch_pixels); print("Channel intensity:", mean, std)
                x, y= plot_gaussian(mean, std, all_ch_pixels, res=80); plt.plot(x, y)  # plot on graph here
            plt.gca().set_xlim(left=0); ax.legend(avail_chs)
            return fig, summary_df
        else:
            return None, summary_df
    
    def get_spotthres(self, summary_df:pd.DataFrame, chs:list):
        ''' 
        Return final threshold of each channel
        Summary_df: Use FULL!!!!!!
        '''
        all_thresholds = dict()
        for ch in chs:
            if self.use_montage:   # Settings
                print("[INFO]Performing auto-threshold by channel:",ch)
                all_thresholds[ch], _ = self.solve_spot_thres_montageIm(
                    collect_channel_pixel(summary_df, ch=ch), k=3)
                print("[Process]Auto-threshold by Montage:", all_thresholds[ch])
            else: all_thresholds[ch] = [0, np.inf]
        return all_thresholds
    
    def solve_spot_thres_montageIm(self, im, k=3, plotful=False):
        #print("[INFO]Collage image(montage) size:",im.shape)
        peak_coords = peak_local_max(im, min_distance=15)
        if plotful:
            fig, ax = plt.subplots(figsize=(50, 100))
            ax_im = ax.imshow(im.astype(np.uint8), cmap='gray')
            ax.plot(peak_coords[:,1], peak_coords[:,0], 'or', markersize=2)
            fig.colorbar(ax_im, ax=ax)
        spots = im[peak_coords[:,0], peak_coords[:,1]]
        if len(spots)<= 1: 
            fit_input=im; print("[WARNING]There are not spots found")
        else: fit_input=spots; print("[INFO] Using",len(spots),"maximas.")
        print('[INFO] Montage mean/min/max:', fit_input.mean(), fit_input.min(), fit_input.max())
        labels, means, covs,_ = gmm(fit_input, k=k)
        means, covs = list(zip(*sorted(zip(means, covs)))); means=list(means)
        stds = []
        for l in range(k):  # Do for each label
            intensities = fit_input[np.where(labels==l)]
            stds.append(np.std(intensities))
        roots =[]; comb = list(it.combinations(range(len(stds)-1),2 ))
        means = np.array(means); stds = np.array(stds)
        for (i1, i2) in comb:
            roots.append(solve_gaussian(means[[i1, i2]], stds[[i1, i2]]))
        # Expect number of pairs of root to be k-1, BUT MAY DEVIATE
        roots = np.array(roots); sorted_roots = roots[np.argsort(roots.max(axis=1))]
        if len(roots)==1:
            low_bound = sorted_roots.min(); high_bound = sorted_roots.max()
        else:
            low_bound = sorted_roots[0].max(); high_bound = sorted_roots[-1].min()
        ## Handle exception of the roots
        if low_bound == high_bound:
            low_bound = min(0, low_bound); high_bound = high_bound
        return (low_bound, high_bound), roots        

    def find_filter_spot(self, im, gaussian, spot_thres =(0, np.inf)):
        print("[INFO] Find & filter spots: Image stats(median/min/max):", np.median(im), im.min(), im.max())
        label_img_original, means, covs, l = gaussian
        bg_label, fg_label = l; spot_label = max(fg_label, bg_label)+1
        labels = {'bg': bg_label, 'fg':fg_label, 'spot':spot_label}
        print("[PROCESS]Labels:", labels)
        #label_img = diameter_closing(label_img, diameter_threshold=3**2)   # close holes
        ## Find peaks
        coords_peak = peak_local_max(im, min_distance=20)
        
        label_img = label_img_original.copy()
        # Filter those belonging to "fg"
        qualifying_spot_coord = coords_peak[np.where(label_img_original[coords_peak[:,0], coords_peak[:,1]] == labels['fg'])]
        print("[PROCESS]Number of Initial Local Maxima:", len(coords_peak))
        print("[PROCESS]Number of foreground qualifying spot",len(qualifying_spot_coord))
        label_img[qualifying_spot_coord[:,0], qualifying_spot_coord[:,1]] = labels['spot']  # [qualifying_spot_loc]
        #print("[PROCESS]Qualifying spots:", len(qualifying_spot_loc),"\nRemained:", label_img[np.where(label_img==labels['spot'])])
        # Montage Gaussian Spots filtering
        def low_intensity_filter(im):
            return np.where(im<np.min(spot_thres))
        def high_intensity_filter(im):
            return np.where(im>np.max(spot_thres))
        label_img_ = label_img.copy()
        # 1. Montage Filter stage (spot too low or high)
        spot_low_coord = qualifying_spot_coord[low_intensity_filter(im[qualifying_spot_coord[:,0],qualifying_spot_coord[:,1]])]
        label_img[spot_low_coord[:,0], spot_low_coord[:,1]] = labels['bg']
        spot_high_coord = qualifying_spot_coord[high_intensity_filter(im[qualifying_spot_coord[:,0],qualifying_spot_coord[:,1]])]
        label_img[spot_high_coord[:,0], spot_high_coord[:,1]] = labels['fg']
        print("[PROCESS]Spots filtered by threshold", len(label_img[np.where((label_img-label_img_)!=0)]),
                "|Remained", len(label_img[np.where(label_img==labels['spot'])]))
        return label_img, labels
    
    def bgsub_calcmax(self, intensities, summary_df, img_name, shifts, gaussian):
        im = plt.imread(summary_df.loc[img_name]["Path"])
        # Perform Image Background subtraction & spot filtering
        im_sub, (cyc_xshift, cyc_y_shift) = subtract_bg(shifts[0], shifts[1], 
                        cyc_img_path=summary_df.loc[img_name]["Bg-Path"], bg_img_path=summary_df.loc[img_name]["Path"])
        print("[INFO]Image shape after subtraction:", im_sub.shape)
        print("[INFO]Subtracted Image Stats:", im_sub.mean(), im_sub.min(), im_sub.max())
        gaussian = gmm(im_sub, k=2)
        # Filter spots: ONLY lower bound (NO upper bound)
        
        label_img, labels = self.find_filter_spot(im_sub, gaussian=gaussian, spot_thres=(np.median(im_sub),np.inf))
        intensities["(BG-sub)Spots"] = im[np.where(label_img==labels['spot'])].ravel()
        intensities["(BG-sub)Bg"] = im[np.where(label_img==labels['bg'])].ravel()
        
        # An original but cropped (due to shift correction)
        im_crop = im[cyc_xshift[0]:im.shape[0]+cyc_xshift[1], cyc_y_shift[0]:im.shape[0]+cyc_y_shift[1]]
        # Write bg_subtracted crop image
        if self.output_bgsub: 
            print("[INFO]Saved background image:",(img_name+'.tif'))
            cv2.imwrite(str(self.bgsub_output_path/(img_name+'.tif')), im_sub)
        return intensities

    def calc_max(self, summary_df:pd.DataFrame, img_name:str, shifts:tuple, spot_thres:tuple, k=2):
        '''
        Take a single FOV and return "spot" intensity & "non-spot" intensity, as determined by
        gaussian mixture model

        Keyword arguments:
        img_name: FOV name
        k: guassian mixture model number of clusters
        diameter threshold: what diameter is a standalone (false) spot
        '''
        intensities = {k:[] for k in ["Bg", "Fg", "Spots"]}
        im = plt.imread(summary_df.loc[img_name]["Path"])
        label_img, means, covs, l = gmm(im, k=2)

        if summary_df.loc[img_name]["Bg-Path"] and self.use_bg:   ######  Background-subtracted Image
            intensities = self.bgsub_calcmax(intensities, summary_df, img_name, shifts, (label_img, means, covs, l ))
        ## Find+filter spots
        label_img_selfthres = label_img.copy()
        # Self Gaussian filtering: 
        means = [np.mean(im[np.where(label_img==i)]) for i in l]
        stds = [np.std(im[np.where(label_img==i)]) for i in l]
        intercepts = solve_gaussian(means, stds).ravel()
        print("[INFO]Foreground/Background Intercepts", intercepts)
        if not len(intercepts)== 1:
            if means[0] >= min(intercepts): midpoint=max(intercepts)
            elif means[1] <= max(intercepts): midpoint=min(intercepts)
            elif (means[0] >= min(intercepts) and means[1] <= max(intercepts)):
                midpoint=np.mean(intercepts)
        print("[INFO] Chosen Fg/Bg midpoint:",midpoint)
        print("[Process] Filtering by self-thresholding: Low bound",midpoint)
        label_img_selfthres, labels_ = self.find_filter_spot(
            im=im, gaussian=(label_img, means, covs, l), spot_thres=(midpoint, np.inf))
        print("[Process] Filtering by Montage-defined thresholding: Low bound", spot_thres[0],"|High bound",spot_thres[1])
        # Then, montage-based thresholding
        label_img_thres, labels_ = self.find_filter_spot(
            im=im, gaussian=(label_img, means, covs, l), spot_thres=spot_thres)
        intensities["Spots"] = im[np.where(label_img_selfthres==labels_['spot'])].ravel()
        intensities["Fg"]= im[np.where(label_img_selfthres==labels_['fg'])].ravel()
        intensities["Bg"]= im[np.where(label_img_selfthres==labels_['bg'])].ravel()
        if self.use_montage:
            intensities["(M)Spots"]= im[np.where(label_img_thres==labels_['spot'])].ravel()
        return intensities

    def calc_S2N(self, intensities):
        def rms(arr): return np.mean(np.array(arr))
        comment=""
        for n in ["Spots","Fg","Bg"]:
            if len(intensities)==0: comment+= "Warning: No pixel belonged in "+n+"\n"
        fg_mean = rms(intensities["Fg"])
        bg_mean = rms(intensities["Bg"])
        fg_snr = fg_mean/bg_mean
        spot_mean = rms(intensities["Spots"])
        spot_fg_snr = spot_mean/fg_mean
        spot_bg_snr = spot_mean/bg_mean
        if self.use_montage:
            mspot_mean = rms(intensities["(M)Spots"])
            mspot_fg_snr = mspot_mean/fg_mean
            mspot_bg_snr = mspot_mean/bg_mean
        else: mspot_mean = mspot_fg_snr = mspot_bg_snr = None
        if self.use_bg:
            bgsub_spot_mean = rms(intensities["(BG-sub)Spots"])
            bgsub_bg_mean = rms(intensities["(BG-sub)Bg"])
            bgsub_sbr = bgsub_spot_mean/bgsub_bg_mean
        else:
            bgsub_spot_mean= bgsub_bg_mean=bgsub_sbr=None
        #std = np.std(intensities[label_name])
        all_info = {"Spot mean":spot_mean, "Spot-fg SNR":spot_fg_snr, "Spot-bg SNR":spot_bg_snr,
            "(M)Spot-fg SNR":mspot_fg_snr, "(M)Spot-bg SNR":mspot_bg_snr, "Fg-Bg SNR":fg_snr, 
            "(M)Spot mean": mspot_mean, "Fg mean":fg_mean, "Bg mean":bg_mean,
            '(BG-sub)Spot mean':bgsub_spot_mean,'(BG-sub)Bg mean':bgsub_bg_mean,
            '(BG-sub)SBR':bgsub_sbr, 'Comment':comment}
        for k,v in all_info.items(): 
            if isinstance(all_info[k],int) or isinstance(all_info[k],float): all_info[k] = np.round(v,2)
        return all_info

    
    def summarize_snr_for_directory(self, cyc_path: Path, fn:str = "AcqData"):
        '''
        This function is the pipline for end-to-end analysis
        use_montage: user set to use montage
        '''
        img_paths = list_img_path(cyc_path/fn)
        _, cyc_summary_df = self.quick_analyze2_by_channel(img_paths = img_paths)
        summary_df_fullcyc = cyc_summary_df.copy()
        if self.FOVs:   # User defined
            print("[INFO] Using user-defined FOVs:",self.FOVs)
            cyc_summary_df = cyc_summary_df.loc[[i for i in self.FOVs if i in cyc_summary_df.index]]
            if len(cyc_summary_df)== 0: 
                print("[Warning]No user-defined FOVs found this cycle.", cyc_summary_df.index)
                return cyc_summary_df
        else:  # Random Sampling
            if self.num_ranSamples and self.num_ranSamples<len(img_paths):
                cyc_summary_df = cyc_summary_df.loc[random.sample(list(cyc_summary_df.index), self.num_ranSamples)]
            print("[Info]Randomly select",len(cyc_summary_df),"for analysis.")
        print("[Info]Total", len(cyc_summary_df),"images for analysis.")
        # Spot thres
        spot_thres_by_channel = self.get_spotthres(summary_df_fullcyc, chs = cyc_summary_df["Channel"].unique())
        print("[Process]Using threshold filter:",
              [(k+':'+str(v)) for k,v in spot_thres_by_channel.items()],sep='\n')
        # Background subtration
        if self.use_bg: cyc_shifts = find_shift(summary_df_fullcyc)
        else: cyc_shifts = None

        for img_name in cyc_summary_df.index:   # Calculate intensities for each image by GMM
            intensities = self.calc_max(summary_df=cyc_summary_df, 
                                        spot_thres=spot_thres_by_channel[cyc_summary_df.loc[img_name,'Channel']],
                                        img_name = img_name, shifts=cyc_shifts)
            #print("[INFO]Intensities:",intensities)
            info = self.calc_S2N(intensities); info["Shift"]=str(cyc_shifts)
            cyc_summary_df.loc[img_name, info.keys()] = pd.Series(info)
            cyc_summary_df.to_csv(cyc_path/("SNR-summary_"+self.cyc_path.name+".csv"))
        return cyc_summary_df

    def summarize_snr_for_all(self):
        cyc_paths = [self.exp_dir/cyc_fp for cyc_fp in os.listdir(self.exp_dir) if 
                       (cyc_fp.startswith("Cyc") and (self.exp_dir/cyc_fp).is_dir)]
        print("[Info]Found cycle folders", cyc_paths)
        sum_dfs = []
        if self.single_run:
            print("[Info]Use single run as in 'AcqData' level.")
            cyc_paths = [self.exp_dir]
        for cyc_path in cyc_paths:   # Iterate folder
            if self.output_bgsub:   # Make bg-subtracted image folder
                self.bgsub_output_path =cyc_path/self.bgsub_fn
                if not os.path.exists(self.bgsub_output_path):
                    os.mkdir(self.bgsub_output_path)
                print("[INFO]Bg-subtracted Images saving to:",self.bgsub_output_path)
            print("[Info] Performing for cycle:",cyc_path.name)
            # Analyze per cycle
            if str(1) not in cyc_path.name:
                self.use_bg = False
            sum_dfs.append(self.summarize_snr_for_directory(cyc_path))
            
        summary_df = pd.concat(sum_dfs, axis=0)
        summary_df = summary_df[['Path','Channel','Bg-Path', 'Mean', 'Std',
                'Fg mean', 'Bg mean', 'Fg-Bg SNR', 'Spot mean', 'Spot-fg SNR', 'Spot-bg SNR',
                '(M)Spot mean', '(M)Spot-fg SNR', '(M)Spot-bg SNR',
                '(BG-sub)Spot mean','(BG-sub)Bg mean','(BG-sub)SBR','Shift','Comment']]
        return summary_df

    def main(self):
        print("[Info]provided experiment path, finding files.")
        self.summarize_snr_for_all().to_csv(self.exp_dir/("SNR-summary_all_"+self.exp_dir.name+".csv"))
        print("[Info]Summary file saved to:", self.exp_dir/("SNR-summary_all_"+self.exp_dir.name+".csv"))


if __name__ == "__main__":
    config = configparser.ConfigParser()

    WORKING_DIR = os.getcwd()
    file_path = Path(os.path.realpath(__file__))

    assert os.path.exists(file_path.parent/"SNRA-settings.ini")
    config.read(file_path.parent/"SNRA-settings.ini")
    if "User Settings" not in config.sections(): 
        settings = config["Default"]; print("[Settings]Using default Settings")
    else: settings = config["User Settings"]; print("User Settings")
    print("[INFO]Experiment folder:",file_path.parent)
    snr_analysis = SNRAnalysis(exp_path=file_path.parent, settings=settings)
    snr_analysis.main()

# %%