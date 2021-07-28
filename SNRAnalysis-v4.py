#!/usr/bin/env python
# coding: utf-8

# In[48]:

import os,  numpy as np, random, skimage.io, matplotlib.pyplot as plt
from skimage.filters import rank,  threshold_local, threshold_otsu
from skimage.morphology import disk
from skimage.feature import peak_local_max
import numpy as np, scipy.ndimage as ndi , PIL.Image as pil, cv2, pandas as pd
import argparse
from pathlib import Path
import matplotlib
from skimage.morphology import area_closing, closing, diameter_closing, remove_small_objects
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from skimage.color import label2rgb
import random
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

def cross_image(p1: Path, p2: Path):
    """
    return (y_shift, x_shift) of fn1 relative to fn2
    """
    im1 = cv2.imread(str(p1), -1).astype('float')
    im2 = cv2.imread(str(p2), -1).astype('float')
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
                            (i.endswith("L473S") and i.contains(str(cyc_num)+"R"))]]
    num_of_registers = max(1, int(len(reg_rows)*0.2)); print("Registering with ", num_of_registers,"FOVs.")
    reg_shifts = []
    for i in range(max(1, num_of_registers)):  # pick 5 fov
        try:
            reg_row = reg_rows.iloc[i]
        except: 
            reg_row = summary_df.iloc[i]
        reg_shifts.append(cross_image(reg_row["Bg-Path"], reg_row["Path"]))
    x_shs = [reg_shifts[i][0] for i in range(len(reg_shifts))]
    y_shs = [reg_shifts[i][1] for i in range(len(reg_shifts))]
    x_sh = int(np.median(x_shs)); y_sh = int(np.median(y_shs))
    # Do image shifting and saving
    if (int(x_sh) | int(y_sh)) == 0:
        print("No shift!")
    else:
        print("Using shift", x_sh, y_sh)
    return x_sh , y_sh

def subtract_bg(x_sh:int, y_sh:int, cyc_img_path:Path, bg_img_path:Path):
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
    return shifted_cyc_img - shifted_bg_img
    
def quick_analyze_no_plot(img_paths:list, bg_path:Path=""):
    fig, ax= plt.subplots()
    img_legends=[]
    summary_df = pd.DataFrame(columns=["Img","Path","Bg-Path", "Mean", "Std",
                'Spot-fg SNR', 'Spot-bg SNR', 'Fg-Bg SNR', 'Spot mean', 'Fg mean', 'Bg mean'])
    for img_path in img_paths:
        img_name_with_ext = img_path.name
        fov_name = img_path.stem
        if "DAPI" in fov_name:
            continue
        if len(str(bg_path))>0:   # Set background path
            if os.path.exists(bg_path/img_name_with_ext): bg_img_path = bg_path/img_name_with_ext
        else: bg_img_path = None

        img_legends.append(fov_name)
        img = cv2.imread(str(img_path))
        mean, std = norm.fit(img.ravel())
        #x, y= plot_gaussian(mean, std, img, res=100)
        summary_df= summary_df.append({"Img":img_path.parent.parent.name+"_"+fov_name, "Path":img_path,
                                        "Bg-Path": bg_img_path, "Mean":mean,"Std":std},
                                      ignore_index=True)
    summary_df.set_index("Img", inplace=True)
    return fig, summary_df

def gmm(im, k=2):
    gm = GaussianMixture(n_components=k, random_state=0)
    labels = gm.fit_predict(im.ravel().reshape(-1,1)).reshape(im.shape)
    return labels, gm.means_.ravel(), gm.covariances_.ravel()

def solve_gaussian(m, std):
    '''Solve Gaussian intersect
    m: list of means, std: list of stds
    '''
    assert len(m)==len(std)
    a = 1/(2*std[0]**2) - 1/(2*std[1]**2)
    b = m[1]/(std[1]**2) - m[0]/(std[0]**2)
    c = m[0]**2 /(2*std[0]**2) - m[1]**2 / (2*std[1]**2) - np.log(std[1]/std[0])
    return np.roots([a,b,c])

def solve_spot_thres_no_plot(montage_path, k=2):
    print("Reading image", montage_path)
    im = cv2.imread(str(montage_path),cv2.IMREAD_UNCHANGED)
    im[np.where(im==im.max())] = 0; im[np.where(im==1000)] = 0
    peak_coords = peak_local_max(im, min_distance=20)
    #fig, ax = plt.subplots(figsize=(50, 100))
    #ax_im = ax.imshow(im.astype(np.uint8), cmap='gray')
    #ax.plot(peak_coords[:,1], peak_coords[:,0], 'or', markersize=2)
    #fig.colorbar(ax_im, ax=ax)
    spots = im[peak_coords[:,0], peak_coords[:,1]]
    labels, means, covs = gmm(spots, k=2)
    stds = []
    for l in range(k):  # Do for each label
        intensities = spots[np.where(labels==l)]
        stds.append(np.std(intensities))
    return solve_gaussian(means, stds)

def calc_max(summary_df, shifts, spot_thres=[0, np.inf], img_name="AN95_L532S", k=2, use_bg=True):
    '''
    Take a single FOV and return "spot" intensity & "non-spot" intensity, as determined by
    gaussian mixture model
    
    Keyword arguments:
    img_name: FOV name
    k: guassian mixture model number of clusters
    diameter threshold: what diameter is a standalone (false) spot
    '''
    # Display "spots" information for single instance
    im = plt.imread(summary_df.loc[img_name]["Path"])
    if summary_df.loc[img_name]["Bg-Path"] and use_bg and shifts:   # Perform Image Background subtraction processing
        im = subtract_bg(shifts[0], shifts[1], 
                        cyc_img_path=summary_df.loc[img_name]["Bg-Path"], bg_img_path=summary_df.loc[img_name]["Path"])
        print("Image shape after subtraction:", im.shape)
        im = im.ravel()[im.ravel()>=0]
    label_img, means, covs = gmm(im, k=k)
    label_img = diameter_closing(label_img, diameter_threshold=5**2)   # close holes
    sorted_labels = np.argsort(means)
    if k==3:
        bg_label, fg_label, spot_label = sorted_labels  # It's not actually spot_label
    elif k==2: 
        bg_label, fg_label = sorted_labels
        spot_label = fg_label+1
    coordinates_unfiltered = peak_local_max(im, min_distance=15)
    coordinates_filtered = np.zeros((0, coordinates_unfiltered.shape[1]))
    intensities = {k:[] for k in ["Spots", "Bg", "Fg"]}
    for coord in coordinates_unfiltered:
        pixel_label = label_img[coord[0],coord[1]] 
        intensity = im[coord[0],coord[1]] 
        if pixel_label != bg_label and intensity<np.max(spot_thres):
            label_img[coord[0],coord[1]] = spot_label
            coordinates_filtered = np.vstack((coordinates_filtered, coord))
    intensities["Fg"]= im[label_img==fg_label].ravel()
    intensities["Bg"]= im[label_img==bg_label].ravel()
    intensities["Spots"]= im[label_img==spot_label].ravel()
    print("Spots filtered", len(coordinates_unfiltered)-len(coordinates_filtered))
    for n, label in enumerate(sorted_labels):
        if label==bg_label: label_name="Bg" 
        elif label==fg_label: label_name="Fg"
        else: label_name="Spots"
        std = np.std(intensities[label_name])
    return {"Spots":intensities["Spots"], "Bg":intensities["Bg"], "Fg":intensities["Fg"]}

def calc_S2N(intensities):
    def rms(arr):
        return np.mean(np.array(arr))
    spot_mean = rms(intensities["Spots"])
    fg_mean = rms(intensities["Fg"])
    bg_mean = rms(intensities["Bg"])
    fg_snr = fg_mean/bg_mean
    spot_fg_snr = spot_mean/fg_mean
    spot_bg_snr = spot_mean/bg_mean
    return {"Spot-fg SNR":spot_fg_snr,"Spot-bg SNR":spot_bg_snr, "Fg-Bg SNR":fg_snr, 
           "Spot mean": spot_mean,"Fg mean":fg_mean, "Bg mean":bg_mean}

def get_names_by_channel(img_paths : list, channels:list):
    imgs = []
    for channel in channels:
        imgs += [p.stem for p in img_paths if p.stem.endswith(channel+"S")]
    return imgs
   

def summarize_snr_for_directory(montage_path=False, img_dir=WORKING_DIR, bg_path:Path="",
                                num_sample=False, fovs:list=[]):
    '''
    This function is the pipline for end-to-end analysis
    montage_path: path to montage if provided
    '''
    img_paths = list_img_path(img_dir)
    fig, summary_df = quick_analyze_no_plot(img_paths = img_paths, bg_path=bg_path)
    imgs_to_use = [img_dir.parent.name+"_"+p.stem for p in img_paths]
    if montage_path:
        channel = montage_path.stem.split("_")[0]; print("Using channel:", channel)
        # Select subset of images
        imgs_to_use = [p for p in imgs_to_use if p.endswith(channel+"S")]
        spot_thres = solve_spot_thres_no_plot(montage_path)
        print("Auto-threshold", spot_thres)
    else: spot_thres = [0, np.inf]
    if len(fovs)>0:
        imgs_to_use = fovs
    if num_sample and num_sample<len(imgs_to_use):
        #print(imgs_to_use[:2],"...")
        imgs_to_use = random.sample(imgs_to_use, num_sample)    
        print("Randomly select",len(imgs_to_use),"for analysis.")
    print("Using", len(imgs_to_use),"images for analysis.")
    try:
        summary_df = summary_df.loc[imgs_to_use]
    except:
        print("Some of the FOVs provided are not found! Ending...")
        print(imgs_to_use); quit()
    if len(str(bg_path))>0:
        print("SUMMARY!", summary_df)
        shifts = find_shift(summary_df); print("Shift calculated:", shifts)
    else:
        shifts = None
    for img_name in summary_df.index:
        intensities = calc_max(summary_df=summary_df, spot_thres=spot_thres, img_name = img_name, k=2,
                                shifts=shifts)
        info = calc_S2N(intensities)
        summary_df.loc[img_name, info.keys()] = pd.Series(info)
    return summary_df

def summarize_snr_for_directory_all_channels(cyc_dir: Path, bg_dir: Path, num_sample: int, fovs:list):
    print("[INFO] Finding Montage Paths in:", cyc_dir)
    montage_paths = [cyc_dir/f for f in os.listdir(cyc_dir)
                     if (cyc_dir/f).is_file() and f.endswith("Montage.tif")]
    sum_dfs = []; num_channel = len(montage_paths)
    if num_channel >= 3 :
        print("[INFO] Found montages", montage_paths)
        print("[INFO] Auto-threshold enabled using montage.")
        for montage_path in montage_paths:
            sum_dfs.append(summarize_snr_for_directory(montage_path, img_dir=cyc_dir/"AcqData", 
                            bg_path = bg_dir, num_sample= num_sample, fovs = fovs))
        summary_df = pd.concat(sum_dfs, axis=0)
    else:
        print("[INFO] None or insufficient montages found!: ", montage_paths)
        print("[INFO] No auto-threshold: high intensity signals will NOT be removed")
        summary_df = summarize_snr_for_directory(False, img_dir=cyc_dir/"AcqData", num_sample= num_sample,
                                                bg_path = bg_dir, fovs = fovs)    
    return summary_df

def summarize_snr_for_all(exp_dir: Path, bg_dir:Path, num_sample: int, fovs:list):
    cyc_folders = []
    for fp in os.listdir(exp_dir):
        if fp.startswith("Cyc") and (exp_dir/fp).is_dir: # fp is cycle folder
            cyc_folders.append(exp_dir/fp)
    print("Found cycle folders", cyc_folders)
    sum_dfs = []
    
    for cyc_folder in cyc_folders:
        sum_dfs.append(summarize_snr_for_directory_all_channels(cyc_folder, bg_dir, num_sample, fovs))
    summary_df = pd.concat(sum_dfs, axis=0)
    return summary_df

def main(exp_dir: Path, bg_dir: Path, settings: dict):
    num_sample = int(settings.get("NumberSampled", 10))
    fovs = settings.get("FOV", "")
    single = bool(int(settings.get("Single", 0)))
    if len(fovs)>0: 
        fovs = fovs.split(','); print("User provided FOVs.")
    if not single:  # provided experiment path
        print("provided experiment path, finding files.")
        summarize_snr_for_all(exp_dir, bg_dir, num_sample=num_sample, fovs=fovs).to_csv(exp_dir/"SNR-summary_all.csv")
    else:                       # AcqData folder level (1 folder only)
        acq_dir = exp_dir
        summarize_snr_for_directory_all_channels(cyc_dir=acq_dir, num_sample=args.n_sample, fovs=fovs).to_csv(
            acq_dir/"SNR-summary_cycle.csv")


if __name__ == "__main__":
    config = configparser.ConfigParser()

    WORKING_DIR = os.getcwd()
    file_path = Path(os.path.realpath(__file__))
    ap = argparse.ArgumentParser(prog='Traditional Cell Segmentation, input image, output mask', add_help=True)
    ap.add_argument("-img","--image_path", help= "Path to acquisition directory (AcqData)", required=False, type = str, default ="")
    ap.add_argument("-exp","--exp_path", help = "Path (parent) to experiment folder (parent folder)", required=False, type = str, default="")
    ap.add_argument("-bg","--bg_path", help = "Name to background folder", required=False, type = str, default="bg")
    args = ap.parse_args()

    if os.path.exists(file_path.parent/"SNRA-settings.ini"):
        config.read(file_path.parent/"SNRA-settings.ini")
        if "User Settings" not in config.sections(): settings = config["Default"]; print("Default Settings")
        else: settings = config["User Settings"]; print("User Settings")

    if os.path.exists(file_path.parent/args.bg_path):
        print("[INFO]Background folder found. Will be used for background subtraction")
        bg_path = file_path.parent/args.bg_path/"AcqData"

    if len(args.exp_path) == 0:
        print("[INFO]Using default experimental path")
        args.exp_path = file_path.parent

    main(exp_dir= Path(args.exp_path), bg_dir = bg_path, settings = settings)