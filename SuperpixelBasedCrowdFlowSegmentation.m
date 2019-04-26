clc;
clear all;
close all;

%%%%%%%%%MotionVectors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
inPath = '/home2/praveen/crowd_ali_db/';
newflnum='crowd002';
mvx = load([inPath,newflnum,'_MVx.mat']);
mvy = load([inPath,newflnum,'_MVy.mat']);
mvx=mvx.MVx_eff;
mvy=mvy.MVy_eff;

mag_checker = zeros(size(mvx,1),size(mvx,2));  
mag_new = sqrt(mvx.^2 + mvy.^2);

frames=size(mag_new,3);

%%%%%%%%%%Camera Compensation%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mv_x = mvx;
% mv_y = mvy;
%  
% num_of_frames = size(mv_x,3);
% frame_x = size(mv_x,2);
% frame_y = size(mv_x,1);
% 
% % mv_y(mv_x > 0.50*max(frame_x,frame_y)) = 0;
% % mv_x(mv_x > 0.50*max(frame_x,frame_y)) = 0;
% % 
% % mv_x(mv_y > 0.50*max(frame_x,frame_y)) = 0;
% % mv_y(mv_y > 0.50*max(frame_x,frame_y)) = 0;
% motion_vec = zeros(frame_y,frame_x,num_of_frames,2);
% %t = [];
% for j = 1:num_of_frames
%     if (nnz(mag_new(:,:,j)) > (0.6*numel(mag_new(:,:,j))))
%      % t = [t j]; 
%         mid_x = ceil(frame_x/2);
%         mid_y = ceil(frame_y/2);
%         [curr_loc_map_x, curr_loc_map_y] = meshgrid((1-mid_x):1:(frame_x-mid_x), (1-mid_y):1:(frame_y-mid_y));
%         curr_motion_x = mv_x(:,:,j);
%         curr_motion_y = mv_y(:,:,j);
%         curr_motion_x = curr_motion_x(:);
%         curr_motion_y = curr_motion_y(:);
%         t1 = (abs(curr_motion_x)>0.5*max(frame_x,frame_y) | abs(curr_motion_y)>0.5*max(frame_x,frame_y));
%         curr_motion_x(t1) = 0;
%         curr_motion_y(t1) = 0;
%         A = zeros(2*numel(curr_motion_x),3);
%         A(1:2:end-1,1) = curr_loc_map_x(:);
%         A(2:2:end,1) = curr_loc_map_y(:);
%         A(1:2:end-1,2) = ones(1,numel(curr_loc_map_x));
%         A(2:2:end,3) = ones(1,numel(curr_loc_map_y));
%         
%         b = zeros(2*numel(curr_loc_map_x),1);
%         next_loc_map_x = curr_loc_map_x(:) + curr_motion_x;
%         next_loc_map_y = curr_loc_map_y(:) + curr_motion_y;
%         
%         b(1:2:end-1) = next_loc_map_x(:);
%         b(2:2:end) = next_loc_map_y(:);
%         
%         coeff(:,j) = A\b;
%         compensation_motion_x = round((coeff(1,j)-1)*curr_loc_map_x(:)+ repmat(coeff(2,j),[numel(curr_loc_map_x) 1]));
%         compensation_motion_y = round((coeff(1,j)-1)*curr_loc_map_y(:)+ repmat(coeff(3,j),[numel(curr_loc_map_y) 1]));
%         
%         motion_vec(:,:,j,1) = reshape(curr_motion_x - compensation_motion_x,frame_y,frame_x,1,1);
%         motion_vec(:,:,j,2) = reshape(curr_motion_y - compensation_motion_y,frame_y,frame_x,1,1);
%         
% %        motion_vec(:,:,j,1) = 0.*reshape(curr_motion_x - compensation_motion_x,frame_y,frame_x,1,1);
% %         motion_vec(:,:,j,2) = 0.*reshape(curr_motion_y - compensation_motion_y,frame_y,frame_x,1,1);
%     else
% %       t = [t j]; 
%         motion_vec(:,:,j,1) = mv_x(:,:,j);
%         motion_vec(:,:,j,2) = mv_y(:,:,j);
%     end
%         
% end
% mvx = squeeze(motion_vec(:,:,:,1));
% mvy = squeeze(motion_vec(:,:,:,2));
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

for frame= 1:frames-4
     median_bunch(:,:,frame) = median(mag_new(:,:,frame:frame+4),3);
end

for i = 1:size(median_bunch,3)
    medianfilteredimage(:,:,i) = medfilt2(median_bunch(:,:,i), [5 5]);
end

for frameCount = 1:size(median_bunch,3)
    mag_checker          = double(medianfilteredimage(:,:,frameCount) > 0) + mag_checker;
end

mag_checker = mag_checker./size(median_bunch,3);
mag_checker(mag_checker <= 0.1) = 0;

mvx = mvx .* repmat(mag_checker,[1 1 size(mvx,3)]);
mvy = mvy .* repmat(mag_checker,[1 1 size(mvy,3)]);


for m=1:size(mvx,1)
    for n=1:size(mvx,2)
        mag = squeeze(sqrt(mvx(m,n,:).^2 + mvy(m,n,:).^2))>0; 
        vx_eff(m,n)=median((mvx(m,n,mag)));

        vy_eff(m,n)=median((mvy(m,n,mag)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%OPtical Flow%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc;
% clear all;
% close all;
% 
% D=dir('/home/manu/Desktop/Flow Segmentation/database');
% D=D(2:8);
% k=3;
% %for i=2:7 
%     video_path='/home/manu/Desktop/Flow Segmentation/database/';
%     %filename=strcat('crowd',int2str(i),'_frames/');
%     readerobj = VideoReader(strcat(video_path,D(k).name));
%     numFrames = readerobj.NumberofFrames;
% 
% %inPath = '/home/manu/Desktop/superpixel with optical flow/OpticalFlow_complex/';
% inPath = '/home/manu/Desktop/optical flow/OpticalFlow/';
% flename = strcat('crowd',int2str(k),'_opticalflow/');
% flowpath=strcat(inPath,flename);
% 
% for i=1:numFrames-1
%    % if (i<10)
%     filename=strcat('OpticalFlow',int2str(i),'.mat');
% %     else
% %      filename=strcat('OpticalFlow00',int2str(i),'.mat');
% %     end
% mvx = load([flowpath,filename]);
% 
% vx=mvx.vx;
% vy=mvx.vy;
% ovx(:,:,i)=vx;
% ovy(:,:,i)=vy;
% % figure(1),quiver(squeeze(vx),squeeze(vy),0)
% % axis ij;
% % title(num2str(i));
% % pause;
% end
% 
% frames=size(ovx,3)
% mag = sqrt(ovx.^2 + ovy.^2);
% mag_mask= (mag > 1);
% ovx = ovx.*mag_mask;
% ovy = ovy.*mag_mask;
% 
% 
%  for m=1:size(ovx,1)
%     for n=1:size(ovx,2)
%         magni = (squeeze(sqrt(ovx(m,n,:).^2 + ovy(m,n,:).^2))>0); 
%         vx_eff(m,n)=median((ovx(m,n,magni)));
% 
%         vy_eff(m,n)=median((ovy(m,n,magni)));
%         end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

magnitude = sqrt(vx_eff.^2 + vy_eff.^2);
magnitude = magnitude +exp(-10);
magnitude_mask = uint8(~isnan(magnitude));
BW_mask = uint8(bwareaopen(magnitude_mask, 100));

figure;
imagesc(BW_mask);

vx_final = vx_eff./(magnitude);
vy_final = vy_eff./(magnitude);
angle_eff=atan2(vy_eff,vx_eff);

final(:,:,1) = vx_final;
final(:,:,2) = vy_final;

result = flowToColor(final);

disp('Entropy Rate Superpixel Segmentation Demo');

filteredimage=medfilt2(rgb2gray(result), [5 5]);

%//=======================================================================
%// Superpixel segmentation
%//=======================================================================
%// nC is the target number of superpixels.

%// Call the mex function for superpixel segmentation\
%// !!! Note that the output label starts from 0 to nC-1.


for i = 3:10
    labels(:,:,i-2) = mex_ers(double(filteredimage),i);
    edgemap(:,:,i-2) = (edge(labels(:,:,i-2),'canny'));
end
BW_finalmask = repmat(BW_mask, [1 1 size(labels,3)]);
labels_final = (uint8(labels+1)).*BW_finalmask;
edgemap_final = (uint8(edgemap)).*BW_finalmask;
edgerate=((sum(edgemap,3)));
edgerate_final = edgerate;

figure;imagesc(edgerate);
conf_score1= edgerate./max(max(edgerate));

dilated_one = bwmorph(edgerate_final,'dilate');
labeled_one = bwlabel(~dilated_one);
labeled_image = bwlabel(edgerate_final);
eroded_one = uint8(bwmorph(dilated_one,'erode'));

confidence_score = zeros(size(labeled_image,1),size(labeled_image,2));
for i=3:size(eroded_one,1)-3
    for j = 3:size(eroded_one,2)-3
        if (eroded_one(i,j) == 1)
            patch = eroded_one(i-2:i+2,j-2:j+2);
            patch_angle = angle_eff(i-2:i+2,j-2:j+2);
            lb = bwlabel(~patch,4);
            lb_one = (lb == 1);
            lb_two = (lb == 2);
            ang_h = sum(sum(lb_one.*patch_angle))./sum(sum(lb_one));
            ang_v = sum(sum(lb_two.*patch_angle))./sum(sum(lb_two));
            
            if (isnan(ang_h)|isnan(ang_v))
                confidence_score(i,j) = pi;
            else
                if (abs(ang_h - ang_v)> pi)
                confidence_score(i,j) = (((2*pi) - (abs(ang_h - ang_v))));
                else
                confidence_score(i,j) = (abs(ang_h - ang_v));
                end
            end
        end
    end
end
conf_score2 = confidence_score./pi;

final_score = conf_score1.*conf_score2;
final_score = uint8(final_score).*BW_mask;

figure;
imagesc(final_score);
title('final_score');

figure;
imagesc(conf_score1);
title('conf_score1');

figure;
imagesc(conf_score2);
title('conf_score2');
final_score = final_score(:)';
zaccard(1,:) = final_score; 
for i = 1 :  size(labels,3)
    zaccard_dist = squeeze(edgemap_final(:,:,i));
    zaccard(2,:) = zaccard_dist(:)';
    dist_measure(i) = 1 - pdist(zaccard,'jaccard')

end

[~,indx] = max(dist_measure);
figure; imagesc(labels_final(:,:,indx));

