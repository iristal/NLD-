function [img_dehazed, transmission] = non_local_dehazing(img_hazy, air_light, gamma,...
    image_name)
%The core implementation of "Non-Local Image Dehazing", CVPR 2016
% 
% The details of the algorithm are described in our paper: 
% Non-Local Image Dehazing. Berman, D. and Treibitz, T. and Avidan S., CVPR2016,
% which can be found at:
% www.eng.tau.ac.il/~berman/NonLocalDehazing/NonLocalDehazing_CVPR2016.pdf
% If you use this code, please cite the paper.
%
%   Input arguments:
%   ----------------
%	img_hazy     - A hazy image in the range [0,255], type: uint8
%	air_light    - As estimated by prior methods, normalized to the range [0,1]
%	gamma        - Radiometric correction. If empty, 1 is assumed
%
%   Output arguments:
%   ----------------
%   img_dehazed  - The restored radiance of the scene (uint8)
%   transmission - Transmission map of the scene, in the range [0,1]
%
% Author: Dana Berman, 2016. 
%
% The software code of the Non-Local Image Dehazing algorithm is provided
% under the attached LICENSE.md


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

img_hazy = im2double(img_hazy);
img_hazy_corrected = img_hazy.^gamma; % radiometric correction

flag_4000 = 0;
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
if flag_4000==0
    n_points = 1000;
    %load pre-calculated uniform tesselation of the unit-sphere
    fid = fopen(['TR',num2str(n_points),'.txt']);
    points = cell2mat(textscan(fid,'%f %f %f')) ;
    fclose(fid);
    mdl = KDTreeSearcher(points);
else
    n_points = 3994;
    load('4000points.mat');
    mdl = KDTreeSearcher(V2);
end
ind = knnsearch(mdl, dist_unit_radius);
used_ind = unique(ind);
% % % dehazed_new = ones(size(dist_unit_radius));
% % % img_hazy_reshaped = reshape(img_hazy_corrected,[h*w,n_colors]);
% % % for i=1:length(used_ind)
% % % %     if i==42
% % % %          pause;
% % % %     end
% % %     temp_ind = find(ind==used_ind(i));
% % %     
% % %     [~, J_current_ind] = max(dist_norm(temp_ind));
% % % %     [~, J_current_ind] = min(abs(dist_norm(temp_ind)-prctile(dist_norm(temp_ind), 90)));
% % %     if isempty(J_current_ind)
% % %         [~, J_current_ind] = max(dist_norm(temp_ind));
% % %     end
% % %     dehazed_new(temp_ind, :) = repmat(img_hazy_reshaped(temp_ind(J_current_ind), :),...
% % %         length(temp_ind), 1);
% % %     D_temp = reshape(dehazed_new, h, w, n_colors); 
% % %     subplot(1, 2, 1);
% % %     imshow(D_temp);
% % %     hold on
% % %     subplot(1, 2, 2);
% % %     imshow(img_hazy_corrected);
% % % %     pause;
% % % end
% % % D = reshape(dehazed_new, h, w, n_colors);    
% % % imshow(D)
% % % title([num2str(n_points), ' color options, ', num2str(length(used_ind)), ' used']);
% % % my_transmission = (bsxfun(@minus, img_hazy_corrected, air_light))./...
% % %     (bsxfun(@minus, D, air_light));
% % % my_transmission = mean(my_transmission, 3);
% % % % figure(1)
% % % % imshow(my_transmission);
%% compare naive transmission vs naive hazelines
%% Estimating Initial Transmission

% Estimate radius as the maximal radius in each haze-line (Eq. (11))
K = accumarray(ind,radius(:),[n_points,1],@max);
radius_new = reshape( K(ind), h, w);
    
% Estimate transmission as radii ratio (Eq. (12))
trans_min = 0.1;
transmission_estimation = radius./radius_new;
% % % img_dehazed = zeros(h,w,n_colors);
% % % leave_haze = 1.06; % leave a bit of haze for a natural look (set to 1 to reduce all haze)
% % % for color_idx = 1:3
% % %     img_dehazed(:,:,color_idx) = ( img_hazy_corrected(:,:,color_idx) - ...
% % %         (1-leave_haze.*transmission_estimation).*air_light(color_idx) )./ max(transmission_estimation,trans_min);
% % % end
% % % 
% % % J_no_reg = img_dehazed;


%% end of compare
% Limit the transmission to the range [trans_min, 1] for numerical stability
trans_min = 0.1;
transmission_estimation = min(max(transmission_estimation, trans_min),1);
before_reg = transmission_estimation;

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
transmission = wls_optimization(transmission_estimation, data_term_weight, img_hazy, lambda);
% % % figure(1)
% % % subplot(2, 2, 1);
% % % imshow(img_hazy_corrected);
% % % subplot(2, 2, 2);
% % % imshow(before_reg); colormap('jet'); title('old T before reg');
% % % subplot(2, 2, 3);
% % % imshow(my_transmission); colormap('jet'); title('new approach T');
% % % subplot(2, 2, 4);
% % % imshow(transmission); colormap('jet'); title('old T after reg');
% saveas(gcf, ['28.5/transmission comp/RESIDE synthetic/', image_name, '.jpg']);


%% Filtering before dehazing
% mask = transmission<0.2;
% [~, img_filtered] = CBM3D(1,img_hazy_corrected.*mask);
% [gradThresh,numIter] = imdiffuseest(rgb2gray(img_hazy_corrected),'ConductionMethod','quadratic');
% img_filtered = imdiffusefilt(img_hazy_corrected,'ConductionMethod','quadratic', ...
%     'GradientThreshold',gradThresh,'NumberOfIterations',numIter);
% img_filtered = imdiffusefilt(img_hazy_corrected);
% img_filtered = medfilt3(img_hazy_corrected, [11, 11, 1]);
% img_hazy_corrected = img_filtered.*mask + img_hazy_corrected.*(1-mask);
%% Dehazing
% (Eq. (16))
img_dehazed = zeros(h,w,n_colors);
leave_haze = 1; % leave a bit of haze for a natural look (set to 1 to reduce all haze)
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
% figure(2)
img_dehazed = im2uint8(img_dehazed);
% % % subplot(2, 2, 1);
% % % imshow(img_hazy_corrected);
% % % subplot(2, 2, 2);
% % % imshow(J_no_reg);  title('old J dehazed before reg');
% % % subplot(2, 2, 3);
% % % % imshow(imread(['images/RESIDE synthetic/original/',image_name, '.jpg']));  title('GT');
% % % % subplot(2, 2, 3);
% % % imshow(D); title('new J dehazed');
% % % subplot(2, 2, 4);
% % % imshow(img_dehazed); title('old J dehazed');
% % % % saveas(gcf, ['28.5/J comp/RESIDE synthetic/', image_name, '.jpg']);
end % function non_local_dehazing
