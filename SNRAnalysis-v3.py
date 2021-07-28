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
import multiprocessing
import matplotlib
from skimage.morphology import area_closing, closing, diameter_closing, remove_small_objects
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from skimage.color import label2rgb
import random
matplotlib.use('TkAgg')

WORKING_DIR = os.getcwd()

def list_img_path(parent_path):
    path_imgs = []
    for file in os.listdir(parent_path):
        if file.endswith(".tif"):
            #print("Is image")
            path_imgs.append(parent_path/file)
    return path_imgs

def plot_gaussian(mean, std, im, res=100):
    x = im.ravel()
    xmin, xmax = np.min(x), np.max(x)
    x = np.linspace(xmin, xmax, res)
    y = norm.pdf(x, mean, std)
    return x, y

def quick_analyze_no_plot(img_paths):
    fig, ax= plt.subplots()
    img_legends=[]
    summary_df = pd.DataFrame(columns=["Img","Path","Mean","Std",
                'Spot-fg SNR', 'Spot-bg SNR', 'Fg-Bg SNR', 'Spot mean', 'Fg mean', 'Bg mean'])
    for img_path in img_paths:
        img_name_with_ext = img_path.name
        fov_name = img_path.stem
        if "DAPI" in fov_name:
            continue
        img_legends.append(fov_name)
        img = cv2.imread(str(img_path))
        mean, std = norm.fit(img.ravel())
        x, y= plot_gaussian(mean, std, img, res=100)
        summary_df= summary_df.append({"Img":img_path.parent.parent.name+"_"+fov_name,"Image":img_name_with_ext,
                                       "Path":img_path,"Mean":mean,"Std":std},
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
    im[np.where(im==im.max())] = 0; im[np.where(im==1000)] = 0;
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

def show_max_no_plot(summary_df, spot_thres=[0, np.inf],img_name="AN95_L532S", k=2):
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
    label_img, means, covs = gmm(im, k=k)
    label_img = diameter_closing(label_img, diameter_threshold=5**2)   # close holes
    sorted_labels = np.argsort(means)
    if k==3:
        bg_label, fg_label, spot_label = sorted_labels  # It's not actually spot_label
    elif k==2: 
        bg_label, fg_label = sorted_labels
        spot_label = fg_label+1
    coordinates_unfiltered = peak_local_max(im, min_distance=20)
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
   

def summarize_snr_for_directory(montage_path=False, img_dir=WORKING_DIR, num_sample=False):
    '''
    This function is the pipline for end-to-end analysis
    montage_path: path to montage if provided
    '''
    img_paths = list_img_path(img_dir)
    fig, summary_df = quick_analyze_no_plot(img_paths = img_paths)
    imgs_to_use = [img_dir.parent.name+"_"+p.stem for p in img_paths]
    if montage_path:
        channel = montage_path.stem.split("_")[0]; print("Using channel:", channel)
        # Select subset of images
        imgs_to_use = [p for p in imgs_to_use if p.endswith(channel+"S")]
        spot_thres = solve_spot_thres_no_plot(montage_path)
        print("Auto-threshold", spot_thres)
    else: spot_thres = [0, np.inf]
    if num_sample:
        #print(imgs_to_use[:2],"...")
        if num_sample > len(imgs_to_use): num_sample = len(imgs_to_use)
        imgs_to_use = random.sample(imgs_to_use, num_sample)
        
    print("Randomly select",len(imgs_to_use),"for analysis.")
    summary_df = summary_df.loc[imgs_to_use]

    for img_name in summary_df.index:
        intensities = show_max_no_plot(summary_df=summary_df, spot_thres=spot_thres, img_name = img_name, k=2)
        info = calc_S2N(intensities)
        summary_df.loc[img_name, info.keys()] = pd.Series(info)
    return summary_df

def summarize_snr_for_directory_all_channels(cyc_dir: Path , num_sample: int):
    print("[INFO] Finding Montage Paths in:", cyc_dir)
    montage_paths = [cyc_dir/f for f in os.listdir(cyc_dir)
                     if (cyc_dir/f).is_file() and f.endswith("Montage.tif")]
    sum_dfs = []; num_channel = len(montage_paths)
    if num_channel >= 3 :
        print("[INFO] Found montages", montage_paths)
        print("[INFO] Auto-threshold enabled using montage.")
        for montage_path in montage_paths:
            sum_dfs.append(summarize_snr_for_directory(montage_path, img_dir=cyc_dir/"AcqData", num_sample= num_sample))
        summary_df = pd.concat(sum_dfs, axis=0)
    else:
        print("[INFO] None or insufficient montages found!: ", montage_paths)
        print("[INFO] No auto-threshold: high intensity signals will NOT be removed")
        summary_df = summarize_snr_for_directory(False, img_dir=cyc_dir/"AcqData", num_sample= num_sample)
    
    summary_df[["Image"]+[c for c in summary_df.columns if c not in ["Image"]]]
    return summary_df

def summarize_snr_for_all(exp_dir: Path, num_sample: int):
    cyc_folders = []
    for fp in os.listdir(exp_dir):
        if fp.startswith("Cyc") and (exp_dir/fp).is_dir: # fp is cycle folder
            cyc_folders.append(exp_dir/fp)
    print("Found cycle folders", cyc_folders)
    sum_dfs = []
    
    for cyc_folder in cyc_folders:
        sum_dfs.append(summarize_snr_for_directory_all_channels(cyc_folder, num_sample))
    summary_df = pd.concat(sum_dfs, axis=0)
    summary_df[["Image"]+[c for c in summary_df.columns if c not in ["Image"]]]
    return summary_df

def main(exp_dir: Path, img_dir: Path, num_sample: int, single_cycle = False):
    
    if not single_cycle:  # provided experiment path
        print("provided experiment path, finding files.")
        summarize_snr_for_all(exp_dir, num_sample=num_sample).to_csv(exp_dir/"SNR-summary_all.csv")
    else:
        acq_dir = exp_dir
        summarize_snr_for_directory_all_channels(cyc_dir=acq_dir, num_sample=args.n_sample).to_csv(
            acq_dir/"SNR-summary_cycle.csv")


if __name__ == "__main__":
    WORKING_DIR = os.getcwd()
    file_path = Path(os.path.realpath(__file__))
    ap = argparse.ArgumentParser(prog='Traditional Cell Segmentation, input image, output mask', add_help=True)
    ap.add_argument("-img","--image_path", help= "Path to acquisition directory (AcqData)", required=False, type = str, default ="")
    ap.add_argument("-exp","--exp_path", help = "Path (parent) to experiment folder (parent folder)", required=False, type = str, default="")
    ap.add_argument("-n","--n_sample", help="number of samples randomly selected (to save time)", default=30, type = int)
    args = ap.parse_args()
    if len(args.exp_path) == 0:
        args.exp_path = file_path.parent
        single_cycle = False
    if "AcqData" in os.listdir(args.exp_path):
        single_cycle = True
    main(exp_dir= Path(args.exp_path), img_dir = Path(args.image_path), num_sample = args.n_sample, single_cycle=single_cycle)   
