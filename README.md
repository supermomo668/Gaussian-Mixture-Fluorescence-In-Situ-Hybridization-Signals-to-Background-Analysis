SNR Analysis

Uses Peak Local Maximum to generate spot candidates then uses Gaussian Mixture Model to generate correct threshold to eliminate spots that are usually false positive due to the foreground (diffuse background + lipofuscid).

[Results]
Generate intensitites for Spots, Foreground, Background. Then also output the SNR ratio for each FOVs. The results is outputted to a spreadsheet (.csv).

[Instructions]
You should be able to drop the "SNR-Analysis-v3.py" (or v4) file in the experimental folder (Along with "Cyc01R", "Cyc02R"...etc) and just double click to run with the "Default" settings)

However, if you want to select your own, you need to use the file "SNRA-settings.ini". Edit the "FOV = " portion under the "User Settings" section and set it to "FOV={FOV_Name1},{FOV_Name2},..." (no space in between) then save. Then, you can double click to run again.

The program can take montage images to help threshold. It can also take background images in "bg" folder to do image subtraction before analyzing.

Changing the code file is not recommended.

[More Info]
The program CAN take montages images and help the thresholding to be more generalized thus more accurate instead of just using the information in a single FOV.

