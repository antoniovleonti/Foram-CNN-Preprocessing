clear;

%%define the parameters for segmentation

% job is a struct; modi, diff, etc. are fields of the struct
job.modi=1; %1 for adaptive segmentation; 2 for traditional bring-up and bring-down
job.diff=1.01; %if adaptive segmentation is used (modi=1), this parameter is ignored <diff means a region with widely-ranged elements, bring-down method will be used
job.con=26; %default for 3D
job.noise=27; %regions with voxels less than this number will be removed
job.cutlarge=false; %default
job.uratio=0.15; job.dratio=1; %only applicable when modi==2;

job.aratio = 0.25; %the fraction parameter
job.minarea = 27000; %regions with voxels less than this number will not be segmented
job.disk1 = strel('disk',1); %creates a "mask" in the shape of a disk
job.close_open = false;

%% filling the grains
load bm;%use the path of the binary image
disp('filling');

%flood fill inner regions
dark_filled     = imclose(dark, job.disk1); %morphological closing
dark_filled     = imfill(dark_filled, 'holes'); 
dark_filled     = imopen(dark_filled, job.disk1); %morophological opening


water.dark      = dark; clear dark;
water.segment   = dark_filled;
water.ridge     = true(size(water.segment));
water.end       = false;

while water.end==false
    water=F_water_main(water,job);
end

segment = water.dark & water.ridge;

implay(segment, 2);

save('segment','segment','job');

