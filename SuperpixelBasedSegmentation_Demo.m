
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

mag_new(:,:,1:29) = mag(:,:,2:30);
mag_new(:,:,30:58) = mag(:,:,32:60);
mag_new(:,:,59:87) = mag(:,:,62:90);
mag_new(:,:,88:107) = mag(:,:,92:111);
frames=size(mag_new,3);
for frame= 1:frames-4
    median_bunch(:,:,frame) = median(mag_new(:,:,frame:frame+4),3);
end

for frameCount = 1:size(median_bunch,3)
	mag_checker = double(median_bunch(:,:,frameCount) > 0) + mag_checker;
end

mag_checker = mag_checker./size(median_bunch,3);
mag_checker(mag_checker <= 0.1) = 0;

mvx = mvx .* repmat(mag_checker,[1 1 size(mvx,3)]);
mvy = mvy .* repmat(mag_checker,[1 1 size(mvy,3)]);


for m=1:size(mvx,1)
    for n=1:size(mvx,2)
        vx_eff(m,n)=median(nonzeros(mvx(m,n,:)));
        vy_eff(m,n)=median(nonzeros(mvy(m,n,:)));
    end
end

magnitude = sqrt(vx_eff.^2 + vy_eff.^2);
vx_final = vx_eff./magnitude;
vy_final = vy_eff./magnitude;

final(:,:,1)=vx_final;
final(:,:,2)=vy_final;

result = flowToColor(final);

disp('Entropy Rate Superpixel Segmentation Demo');

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