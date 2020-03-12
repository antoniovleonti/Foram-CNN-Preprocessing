
%%
function water = F_water_main(water, job)

% WHAT is cutlarge??
if job.cutlarge && min(water.ridge(:)) == 1
    disp('first time!')
end

aratio = job.aratio;

disp('labelling and calculating property')
temp_0 = bwconncomp(water.segment); %connected components

% From https://www.mathworks.com/help/images/ref/labelmatrix.html, on the return value of labelmatrix():
% "Label matrix of contiguous regions, returned as matrix of nonnegative integers. The pixels labeled 0 are the background. The pixels labeled 1 make up one object; the pixels labeled 2 make up a second object; and so on."
% EXAMPLE: (0 is background, 1, is the "first" object, 2 is the second, etc.)
% { 0 0 0 0 0 0 0 0 0 0 ;                  { 0 0 0 0 0 0 0 0 0 0 ;
%   0 1 1 0 1 1 0 1 1 0 ;                    0 1 1 0 2 2 0 3 3 0 ;
%   0 1 1 0 1 1 0 1 1 0 ; ->labelmatrix()->  0 1 1 0 2 2 0 3 3 0 ;
%   0 0 0 0 0 0 0 0 0 0 }                    0 0 0 0 0 0 0 0 0 0 }
label_0 = labelmatrix(temp_0);
% calculate area of each region
label_0p = regionprops(temp_0,'area');
areas = [label_0p.Area];

clear temp_0; clear label_0p;


cri_size = job.minarea;
clear temp;

% t1, t2, and t3 are the dimensions of the data
[t1, t2, t3] = size(water.segment);
% create array of 1s the same size as the data
ridge_1 = true(t1, t2, t3);

% for each region... (indices start at 1 in matlab, not zero -_-)
for counter = 1 : max(label_0(:)) % for var start:end
    % print the counter variable
    disp(counter);

    % if the area of the currently "selected" region
    if (areas(counter) >= cri_size)
        % make a copy of label_0 where the currently selected region is '1' and everything else is 0
        particle = label_0 == counter;

        %% ************** "CROP" DATA **************
        % crop data to only contain cells necessary to represent the currently selected region

        % This returns an array of subscripts, and those subscripts point to every single data point representing the currently selected region
        % This is necessary because find() returns an array of linear indices
        [xx, yy, zz] = ind2sub([t1, t2, t3], find(particle));

        %disp([double(counter), double(max(label_0(:))), double(areas(counter)), double(max(areas))]);

        % so now we get the range of integer values in every direction where something lies at this height (height = "counter")
        xrange = min(xx) : max(xx);  % 0 : x = { 0, 1, 2, ... , x-1, x }
        yrange = min(yy) : max(yy);
        zrange = min(zz) : max(zz);

        % if the currently selected area > .75 * the greatest area
        %if areas(counter) > (0.75 * max(areas))
            % just a print statement
        disp([min(xx), max(xx), min(yy), max(yy), min(zz), max(zz)]);
        %end

        % particle_local is the final, cropped version of the data
        particle_local = particle(xrange, yrange, zrange);
        clear particle;

        %%

        % "local" euclidean distance map is a distance map of the inverted 3d partition of particle ??
        EDM_local = bwdist(~particle_local);
        % we now modify this iedm using our adaptive function
        EDM_local_1 = F_water_modifyedm_adaptive(-EDM_local, aratio);
        clear EDM_local;

        disp('watersheding');
        % using the stock watershed function here ??
        % do these functions "just work" on 3d images???
        water_local = watershed(EDM_local_1);
        % put a 1 where particle_local == 1 and the watershed result == 0
        cutline = particle_local & (water_local == 0);

        ridge_1(xrange, yrange, zrange) = min(ridge_1(xrange, yrange, zrange), ~cutline);

        % remove this particle if it's not segmented this time
        if max(cutline(:)) == 0
            label_0(label_0 == counter) = 0;
        end
        clear water_local; clear cutline;
    else
        label_0(label_0 == counter) = 0;
    end
end

if min(ridge_1) == 1
    water.end = true;
end

if job.modi == 2 %no iteration for the traditional method
    water.end = true;
end

water.ridge = water.ridge & ridge_1;
water.segment = water.segment & (label_0 > 0);%not useful after iteration
water.segment = water.segment & ridge_1;
