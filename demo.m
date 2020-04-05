
image_name = '4';
%%%%
% Load the gamma from the param file. 
% These values were given by Ra'anan Fattal, along with each image:
% http://www.cs.huji.ac.il/~raananf/projects/dehaze_cl/results/
gamma=1.5;


img_hazy_denoised = pre_processing(img_hazy);       
% Estimate air-light using our method described in:
% Air-light Estimation using Haze-Lines. Berman, D. and Treibitz, T. and 
% Avidan S., ICCP 2017
A = reshape(estimate_airlight(im2double(img_hazy_denoised).^(gamma)),1,1,3);

% Dehaze the image	
[img_dehazed, trans_refined] = non_local_dehazing_lb(img_hazy_denoised, A, gamma);
imwrite(img_dehazed,['results/',image_name,'.png']);

