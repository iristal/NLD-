function img_dehazed = dehazing(img_hazy_corrected, air_light, gamma, transmission,trans_min)
% (Eq. (16))
[h,w,n_colors] = size(img_hazy_corrected);
img_dehazed = zeros(h,w,n_colors);
leave_haze = 1.06; % leave a bit of haze for a natural look (set to 1 to reduce all haze)
for color_idx = 1:3
    img_dehazed(:,:,color_idx) = ( img_hazy_corrected(:,:,color_idx) - ...
        (1-leave_haze.*transmission).*air_light(color_idx) )./ max(transmission,trans_min);
end

% Limit each pixel value to the range [0, 1] (avoid numerical problems)
img_dehazed(img_dehazed>1) = 1;
img_dehazed(img_dehazed<0) = 0;
img_dehazed = img_dehazed.^(1/gamma); % radiometric correction

% For display, we perform a global linear contrast stretch on the output, 
% clipping 0.5% of the pixel values both in the shadows and in the highlights 
adj_percent = [0.005, 0.995];
img_dehazed = adjust(img_dehazed,adj_percent);

img_dehazed = im2uint8(img_dehazed);

end
