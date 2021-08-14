Gaussian-Mixture Model-based Fluorescence-In-Situ-Hybridization Signal-to-noise ratio Analysis

# Overview

Uses Peak Local Maximum to generate spot candidates then uses Gaussian Mixture Model to generate correct threshold to eliminate spots that are usually false positive due to the foreground (diffuse background + lipofuscid).

# Results
Generate intensitites for Spots, Foreground, Background. Then also output the SNR ratio for each FOVs. The results is outputted to a spreadsheet (.csv).

# Instructions
You should be able to drop the **SNR-Analysis-v3.1.py** (or v5) file in the experimental folder (Along with "Cyc01R", "Cyc02R"...etc) and just double click to run with the "Default" settings)

However, if you want to select your own, you need to use the file **SNRA-settings.ini**. Edit the **FOV** portion under **User Settings** section and set it to 
```
FOV={FOV_Name1},{FOV_Name2},...
(no space in between) then save. Then, you can double click to run again.
```
The program can use collage images (aggregate all images from a single channel) to help Auto-thresholding to distinguish what is real or false background. It can also take background images in "bg" folder to do image subtraction before analyzing.

Changing the code file is not recommended.

# More Information

The program is experimental but is scientifically sound. It makes use of available information to disguish signals and background (if background is not provided). It can also make use of control background images to make much better estimation of SNR ratios.

An assumption for each image we take is that the intensities profile resembles the following, and that is generally true:
![alt text](https://chrisjmccormick.files.wordpress.com/2014/08/1d_example.png)
If the profile looks like this then we have a foreground and a background. It should actually looks similar because in tissue imaging, you either have tissue or there isn't and their intensity should be distinctive enough to tell most of the time.

Then, local peak maximum is performed to find the signals. When it is performed, it would result in false positives and the next task is to remove them.

Let's say if the entire tissue on a slide (after numerous images have been taken on this tissue slide) and the tissue has intensities profile that looks like the following: we call the entirety of these images "Montage" or "Collage". If settings is set to "use Montage", it will attempt to fit the profile on the montage based on Gaussian Mixture (k=3), again. In our use case, tissue is expected to have distinctive intensities difference between the foreground and the signals. Then, you may expect the profile will look like:
![alt text](https://miro.medium.com/max/1400/1*lTv7e4Cdlp738X_WFZyZHA.png)
We then use the intercepts of these Gaussian to generate threshold.
