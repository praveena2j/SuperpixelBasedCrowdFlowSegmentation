
%%%%%%%%%MotionVectors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clc;
clear all;
close all;
inPath = '/home2/praveen/crowd_cfsas_db/';
newflnum='crowd002';
mvx = load([inPath,newflnum,'_MVx.mat']);
mvy = load([inPath,newflnum,'_MVy.mat']);
mvx=mvx.MVx_eff;
mvy=mvy.MVy_eff;
mag_checker = zeros(size(mvx,1),size(mvx,2));  
mag_check = 255*zeros(size(mvx,1),size(mvx,2));  
mag = sqrt(mvx.^2 + mvy.^2);


% k=1;
% for i = 1: 29 : size(mag,3)
%     if ((i+k+28)<size(mag,3))
%         mag_new(:,:,i:i+28) = mag(:,:,(i+k):(i+k+28));
%         k=k+1;
%     end
% end



mag_new(:,:,1:29) = mag(:,:,2:30);
mag_new(:,:,30:58) = mag(:,:,32:60);
mag_new(:,:,59:87) = mag(:,:,62:90);
mag_new(:,:,88:107) = mag(:,:,92:111);
frames=size(mag_new,3);
 for frame= 1:frames-4
     median_bunch(:,:,frame) = median(mag_new(:,:,frame:frame+4),3);
 end

% for frame= 1:frames-4
%     for i=1:
%     mag_new(i+2)=median(mag_new())
for frameCount = 1:size(median_bunch,3)
    mag_checker          = double(median_bunch(:,:,frameCount) > 0) + mag_checker;
end
  
% for l=1:size(mvx,1)*size(mvx,2)
%     if (mag_checker(l)<0.1*(size(median_bunch,3)))
%         mag_check(l)=0;
%     end
% end

mag_checker = mag_checker./size(median_bunch,3);
mag_checker(mag_checker <= 0.1) = 0;

% for k=1:size(mvx,3)
%     frame_x=mvx(:,:,k);
%     frame_y=mvy(:,:,k);
%     for t=1:size(mvx,1)*size(mvx,2)
%         if(uint8(mag_checker(t))==0)
%             frame_x(t)=0;
%             frame_y(t)=0;
%         end
%     end
%     mvx(:,:,k)=frame_x;
%     mvy(:,:,k)=frame_y;
% end

mvx = mvx .* repmat(mag_checker,[1 1 size(mvx,3)]);
mvy = mvy .* repmat(mag_checker,[1 1 size(mvy,3)]);


for m=1:size(mvx,1)
    for n=1:size(mvx,2)
      %  if(nnz(mvx(m,n,:)>0))
        vx_eff(m,n)=median(nonzeros(mvx(m,n,:)));
%         else
%             mvx_eff(m,n)=0;
%        end
%            if(nnz(mvy(m,n,:)>0))
        vy_eff(m,n)=median(nonzeros(mvy(m,n,:)));
%             else
%                 mvy_eff(m,n)=0;
%             end
        end
end




%for i = 1:size(mvx,3)
%     figure;
%     quiver(squeeze(vx_eff3),squeeze(vy_eff3),0); 
% %     quiver(squeeze(mag_checker)); 
%      axis ij; 
% %    title(num2str(i)); 
%  pause; 
%end


% mag_eff = sqrt(vx_eff3.^2 + vy_eff3.^2);
% 
% figure;
% imagesc(mag_eff);
% title('mag_eff');
% 
% angle_eff = atan2(vy_eff3,vx_eff3);

% figure;
% imagesc(angle_eff);
% title('angle_eff');
% 
% 
% 
% H = fspecial('gaussian',[3 3],0.5);
% blur=imfilter(angle_eff,H,''replicate);
% figure;
% imagesc(blur);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%OPtical Flow%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc;
% clear all;
% close all;
% 
% D=dir('/home/manu/Desktop/Flow Segmentation/database');
% D=D(2:8);
% k=2;
% %for i=2:7 
%     video_path='/home/manu/Desktop/Flow Segmentation/database/';
%     %filename=strcat('crowd',int2str(i),'_frames/');
%     readerobj = VideoReader(strcat(video_path,D(k).name));
%     numFrames = readerobj.NumberofFrames;
% 
% 
% 
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
% %mag_checker = zeros(size(vx,1),size(vx,2));  
% %mag_check = zeros(size(vx,1),size(vx,2));  
% frames=size(ovx,3)
% mag = sqrt(ovx.^2 + ovy.^2);
% mag_mask= (mag > 1);
% ovx = ovx.*mag_mask;
% ovy = ovy.*mag_mask;
% %  for i=1:size(ovx,1)
% %       for j=1:size(ovx,2)
% %           for k=1:size(ovx,3)
% %               if(mag(i,j,k)<0.5)
% %                   ovx(i,j,k)=0;
% %                   ovy(i,j,k)=0;
% %               end
% %           end
% %       end
% %  end
% % magnitude = sqrt(ovx.^2 + ovy.^2);
% % ovx = ovx./magnitude;
% % ovy = ovy./magnitude;
% for m=1:size(vx,1)
%     for n=1:size(vx,2)
%        % if(nnz(vx(m,n,:))>0)
%             vx_eff(m,n)=median(nonzeros(ovx(m,n,:)));        
% %         else
% %             vx_eff3(m,n)=0;        
% %         end
% %        if(nnz(vy(m,n,:))>0)
%             vy_eff(m,n)=median(nonzeros(ovy(m,n,:)));
% %         else
% %             vy_eff3(m,n)=0;        
% %         end
%     end
%end


magnitude = sqrt(vx_eff.^2 + vy_eff.^2);
vx_final = vx_eff./magnitude;
vy_final = vy_eff./magnitude;
%final=zeros(size(vx_final,1),size(final,2),2);
final(:,:,1)=vx_final;
final(:,:,2)=vy_final;

result = flowToColor(final);




% angle_eff=atan2(vy_eff3,vx_eff3);
% figure;
% imagesc(angle_eff);
% 
% 
% H = fspecial('gaussian',[7 7],0.5);
% blur=imfilter(angle_eff,H,'replicate');
% figure;
% imagesc(blur);

% % angle_eff=zeros(size(vx_eff,1),size(vx_eff,2));
% %  for i=1:size(vx_eff,1)*size(vx_eff,2)
% %      if (vx_eff(i)==0 && vy_eff(i)==0)
% %          angle_eff(i) =4;
% %     
% %      else
% %      angle_eff(i)=atan2(vy_eff(i),vx_eff(i));
% %     end
% %  end
% % 
% % angle_eff(isnan(angle_eff)) = 4;
% % g_angle_eff1 = uint8((angle_eff+pi)./(pi+4) * 255);
% % 
% % 
% % g_angle_eff2 = uint8(mod(angle_eff+pi+1,(pi+4))./(pi+4) * 255);
% % 
% % g_angle_eff3 = uint8(mod(angle_eff+pi+2,(pi+4))./(pi+4) * 255);
% % % figure;
% % % imshow(g_angle_eff);
% % 
% % filteredimage1=medfilt2(g_angle_eff1,[ 5 5]);
% % filteredimage2=medfilt2(g_angle_eff2,[ 5 5]);
% % filteredimage3=medfilt2(g_angle_eff3,[ 5 5]);
% % % H = fspecial('gaussian',[3 3],0.5);
% % % blur=imfilter(angle_eff,H,'replicate');
% % % figure;
% % % imshow(filteredimage);
% % % final=zeros(size(filteredimage1,1),size(filteredimage1,2),3);
% % final(:,:,1)=filteredimage1;
% % final(:,:,2)=filteredimage2;
% % final(:,:,3)=filteredimage3;
% % %blur = double(filteredimage);
% % % for i=1:size(g_angle_eff,1)*size(g_angle_eff,2)
% % %     if(g_angle_eff(i) == NaN)
% % %         g_angle_eff(i)=9;
% % %     end
% % % end
% % 

disp('Entropy Rate Superpixel Segmentation Demo');

%%
%//=======================================================================
%// Input
%//=======================================================================
%// These images are duplicated from the Berkeley segmentation dataset,
%// which can be access via the URL
%// http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
%// We use them only for demonstration purposes.

% img = imread('148089.jpg');
%img = imread('242078.jpg');
%img = 50*angle_eff;
%// Our implementation can take both color and grey scale images.
%grey_img = double(rgb2gray(img));
grey_img =blur;
img=grey_img;
%%
%//=======================================================================
%// Superpixel segmentation
%//=======================================================================
%// nC is the target number of superpixels.
nC = 300;
%// Call the mex function for superpixel segmentation\
%// !!! Note that the output label starts from 0 to nC-1.
t = cputime;

lambda_prime = 0.5;sigma = 5.0; 
conn8 = 1; % flag for using 8 connected grid graph (default setting).

[labels] = mex_ers(double(img),nC);
%[labels] = mex_ers(double(img),nC,lambda_prime,sigma);
%[labels] = mex_ers(double(img),nC,lambda_prime,sigma,conn8);

% grey scale iamge
%[labels] = mex_ers(grey_img,nC);
%[labels] = mex_ers(grey_img,nC,lambda_prime,sigma);
%[labels] = mex_ers(grey_img,nC,lambda_prime,sigma,conn8);

fprintf(1,'Use %f sec. \n',cputime-t);
fprintf(1,'\t to divide the image into %d superpixels.\n',nC);

%// You can also specify your preference parameters. The parameter values
%// (lambda_prime = 0.5, sigma = 5.0) are chosen based on the experiment
%// results in the Berkeley segmentation dataset.
%// lambda_prime = 0.5; sigma = 5.0;
%// [labels] = mex_ers(grey_img,nC,lambda_prime,sigma);
%// You can also use 4 connected-grid graph. The algorithm uses 8-connected 
%// graph as default setting. By setting conn8 = 0 and running
%// [labels] = mex_ers(grey_img,nC,lambda_prime,sigma,conn8),
%// the algorithm perform segmentation uses 4-connected graph. Note that 
%// 4 connected graph is faster.


%%
%//=======================================================================
%// Output
%//=======================================================================
[height width] = size(grey_img);

%// Compute the boundary map and superimpose it on the input image in the
%// green channel.
%// The seg2bmap function is directly duplicated from the Berkeley
%// Segmentation dataset which can be accessed via
%// http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
[bmap] = seg2bmap(labels,width,height);
bmapOnImg = img;
idx = find(bmap>0);
timg = grey_img;
timg(idx) = 255;
bmapOnImg(:,:,2) = timg;
bmapOnImg(:,:,1) = grey_img;
bmapOnImg(:,:,3) = grey_img;

%// Randomly color the superpixels
[out] = random_color( double(img) ,labels,nC);

%// Compute the superpixel size histogram.
siz = zeros(nC,1);
for i=0:(nC-1)
    siz(i+1) = sum( labels(:)==i );
end
[his bins] = hist( siz, 20 );

%%
%//=======================================================================
%// Display 
%//=======================================================================
gcf = figure(1);
subplot(2,3,1);
imshow(img,[]);
title('input image.');
subplot(2,3,2);
imshow(bmapOnImg,[]);
title('superpixel boundary map');
subplot(2,3,3);
imshow(out,[]);
title('randomly-colored superpixels');
subplot(2,3,5);
bar(bins,his,'b');
title('the distribution of superpixel size');
ylabel('# of superpixels');
xlabel('superpixel sizes in pixel');
scnsize = get(0,'ScreenSize');
set(gcf,'OuterPosition',scnsize);