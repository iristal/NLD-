function denoised = pre_processing(img_hazy)
sigma = 5;
[~ , denoised] = CBM3D(1, im2double(img_hazy), sigma);
denoised = im2uint8(denoised);
