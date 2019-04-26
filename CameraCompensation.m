mv_x = mvx;
mv_y = mvy;
 
num_of_frames = size(mv_x,3);
frame_x = size(mv_x,2);
frame_y = size(mv_x,1);

% mv_y(mv_x > 0.50*max(frame_x,frame_y)) = 0;
% mv_x(mv_x > 0.50*max(frame_x,frame_y)) = 0;

% mv_x(mv_y > 0.50*max(frame_x,frame_y)) = 0;
% mv_y(mv_y > 0.50*max(frame_x,frame_y)) = 0;
motion_vec = zeros(frame_y,frame_x,num_of_frames,2);
%t = [];
for j = 1:num_of_frames
    if (nnz(mag_new(:,:,j)) > (0.6*numel(mag_new(:,:,j))))
     % t = [t j]; 
        mid_x = ceil(frame_x/2);
        mid_y = ceil(frame_y/2);
        [curr_loc_map_x, curr_loc_map_y] = meshgrid((1-mid_x):1:(frame_x-mid_x), (1-mid_y):1:(frame_y-mid_y));
        curr_motion_x = mv_x(:,:,j);
        curr_motion_y = mv_y(:,:,j);
        curr_motion_x = curr_motion_x(:);
        curr_motion_y = curr_motion_y(:);
        t1 = (abs(curr_motion_x)>0.5*max(frame_x,frame_y) | abs(curr_motion_y)>0.5*max(frame_x,frame_y));
        curr_motion_x(t1) = 0;
        curr_motion_y(t1) = 0;
        A = zeros(2*numel(curr_motion_x),3);
        A(1:2:end-1,1) = curr_loc_map_x(:);
        A(2:2:end,1) = curr_loc_map_y(:);
        A(1:2:end-1,2) = ones(1,numel(curr_loc_map_x));
        A(2:2:end,3) = ones(1,numel(curr_loc_map_y));
        
        b = zeros(2*numel(curr_loc_map_x),1);
        next_loc_map_x = curr_loc_map_x(:) + curr_motion_x;
        next_loc_map_y = curr_loc_map_y(:) + curr_motion_y;
        
        b(1:2:end-1) = next_loc_map_x(:);
        b(2:2:end) = next_loc_map_y(:);
        
        coeff(:,j) = A\b;
        compensation_motion_x = round((coeff(1,j)-1)*curr_loc_map_x(:)+ repmat(coeff(2,j),[numel(curr_loc_map_x) 1]));
        compensation_motion_y = round((coeff(1,j)-1)*curr_loc_map_y(:)+ repmat(coeff(3,j),[numel(curr_loc_map_y) 1]));
        
        motion_vec(:,:,j,1) = reshape(curr_motion_x - compensation_motion_x,frame_y,frame_x,1,1);
        motion_vec(:,:,j,2) = reshape(curr_motion_y - compensation_motion_y,frame_y,frame_x,1,1);
        
%        motion_vec(:,:,j,1) = 0.*reshape(curr_motion_x - compensation_motion_x,frame_y,frame_x,1,1);
%         motion_vec(:,:,j,2) = 0.*reshape(curr_motion_y - compensation_motion_y,frame_y,frame_x,1,1);
    else
%       t = [t j]; 
        motion_vec(:,:,j,1) = mv_x(:,:,j);
        motion_vec(:,:,j,2) = mv_y(:,:,j);
    end
        
end
mvx = squeeze(motion_vec(:,:,:,1));
mvy = squeeze(motion_vec(:,:,:,2));
