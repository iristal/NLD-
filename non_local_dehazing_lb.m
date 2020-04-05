function [img_dehazed, transmission] = non_local_dehazing_lb(img_hazy, air_light, gamma)


%% Validate input
[h,w,n_colors] = size(img_hazy);
if (n_colors ~= 3) % input verification
    error(['Non-Local Dehazing reuires an RGB image, while input ',...
        'has only ',num2str(n_colors),' dimensions']);
end

if ~exist('air_light','var') || isempty(air_light) || (numel(air_light)~=3)
    error('Dehazing on sphere requires an RGB airlight');
end

if ~exist('gamma','var') || isempty(gamma), gamma = 1; end

max_size=[600 600];
flag=0;
if (h*w > max_size(1)*max_size(2))
    flag=1;
    img_hazy_orig = img_hazy;
    
    %%% for upsampling:
    uf=ceil(log2(max(h/max_size(1),w/max_size(2))));
    img_hazy = imresize(img_hazy ,1/(2^uf), 'bicubic');
    
    [h,w,n_colors] = size(img_hazy);
    img_hazy = im2double(img_hazy);
    img_hazy_corrected = img_hazy.^gamma; % radiometric correction
else
    [h,w,n_colors] = size(img_hazy);
    img_hazy = im2double(img_hazy);
    img_hazy_corrected = img_hazy.^gamma; % radiometric correction
    
end 
%% Find Haze-lines
% Translate the coordinate system to be air_light-centric (Eq. (3))
dist_from_airlight = double(zeros(h,w,n_colors));
for color_idx=1:n_colors
    dist_from_airlight(:,:,color_idx) = img_hazy_corrected(:,:,color_idx) - air_light(:,:,color_idx);
end

% Calculate radius (Eq. (5))
radius = sqrt( dist_from_airlight(:,:,1).^2 + dist_from_airlight(:,:,2).^2 +dist_from_airlight(:,:,3).^2 );

% Cluster the pixels to haze-lines
% Use a KD-tree impementation for fast clustering according to their angles
dist_unit_radius = reshape(dist_from_airlight,[h*w,n_colors]);
dist_norm = sqrt(sum(dist_unit_radius.^2,2));
dist_unit_radius = bsxfun(@rdivide, dist_unit_radius, dist_norm);
n_points = 1000;
% load pre-calculated uniform tesselation of the unit-sphere
fid = fopen(['TR',num2str(n_points),'.txt']);
points = cell2mat(textscan(fid,'%f %f %f')) ;
fclose(fid);
mdl = KDTreeSearcher(points);
ind = knnsearch(mdl, dist_unit_radius);


%% Estimating Initial Transmission

% Estimate radius as the maximal radius in each haze-line (Eq. (11))
K = accumarray(ind,radius(:),[n_points,1],@max);
radius_new = max(K)*ones(h,w);     
% Estimate transmission as radii ratio (Eq. (12))
transmission_estimation = radius./radius_new;
imInd=gray2ind(transmission_estimation,256);
jetRGB=ind2rgb(imInd,jet(256));

% Limit the transmission to the range [trans_min, 1] for numerical stability
trans_min = 0.1;
transmission_estimation = min(max(transmission_estimation, trans_min),1);


%% Regularization

% Apply lower bound from the image (Eqs. (13-14))
trans_lower_bound = 1 - min(bsxfun(@rdivide,img_hazy_corrected,reshape(air_light,1,1,3)) ,[],3);
transmission_estimation = max(transmission_estimation, trans_lower_bound);
 
 % Solve optimization problem (Eq. (15))
% find bin counts for reliability - small bins (#pixels<50) do not comply with 
% the model assumptions and should be disregarded
bin_count       = accumarray(ind,1,[n_points,1]);
bin_count_map   = reshape(bin_count(ind),h,w);
bin_eval_fun    = @(x) min(1, x/50);

% Calculate std - this is the data-term weight of Eq. (15)
K_std = accumarray(ind,radius(:),[n_points,1],@std);
radius_std = reshape( K_std(ind), h, w);
radius_eval_fun = @(r) min(1, 3*max(0.001, r-0.1));
radius_reliability = radius_eval_fun(radius_std./max(radius_std(:)));
data_term_weight   = bin_eval_fun(bin_count_map).*radius_reliability;
lambda = 0.1;

transmission = optimization(transmission_estimation, data_term_weight, img_hazy, lambda, trans_lower_bound);
   
if (flag)
    img_hazy = img_hazy_orig;
    [h,w,n_colors] = size(img_hazy);
    img_hazy = im2double(img_hazy);
    img_hazy_corrected = img_hazy.^gamma; % radiometric correction
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  Residual Interpolation for Internsity Guided Depth Upsampling %%%%%%%%%%%%%%%
    scale=uf;
    big=2^uf*size(transmission);
    guide_s=imresize(im2double(rgb2gray(img_hazy)),big);
    guide_s = guide_s/max(guide_s(:));

    for i = 1:scale
        hs = fspecial('gaussian', 2^(scale-i), 2^(scale-i));
        guide_tmp = imfilter(guide_s, hs, 'replicate');
        guide = guide_tmp(1:2^(scale-i):end, 1:2^(scale-i):end);
        transmission_new = resint(transmission, guide);
        img=imresize(img_hazy_corrected,size(transmission_new));
        guide_new = dehazing(img, air_light, gamma, transmission_new,trans_min);
        guide_new=im2double(rgb2gray(guide_new));
        guide_new = guide_new/max(guide_new(:));
        transmission = resint(transmission, guide_new);
        sum(sum(abs(transmission_new-transmission)));
        size(guide_new);
    end
    transmission=imcrop(transmission, [1 1 w-1 h-1]);     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end  

%% Dehazing
% (Eq. (16))
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

end % function non_local_dehazing
