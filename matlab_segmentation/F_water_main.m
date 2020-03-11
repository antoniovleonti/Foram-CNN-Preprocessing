
%%
function water = F_water_main(water,job)

% wtf is cutlarge??
if job.cutlarge && min(water.ridge(:)) == 1
    disp('first time!')
end

aratio = job.aratio;

disp('labelling and calculating property')
temp_0   = bwconncomp(water.segment); %connected components
label_0  = labelmatrix(temp_0);
label_0p = regionprops(temp_0,'area');
areas    = [label_0p.Area];
clear temp_0; clear label_0p;


cri_size = job.minarea;
clear temp;


% t1,t2,and t3 are the dimensions of the slices
[t1,t2,t3]=size(water.segment);
% create array of 1s of size size(water.segment)
ridge_1=true(t1,t2,t3);

% for 1 to the largest item in the array of connected components (whatever that means)
% indices start at 1 in matlab, not zero
for counter = 1:max(label_0(:)) % for var start:end
    % print the counter
    disp(counter);
    % if the currently selected contours area >= our previously set area threshold, ...

    % WHAT DOES THIS IF BLOCK DO??? (in one sentence)
    % it looks for a certain value in the
    if (areas(counter) >= cri_size)
        % "x = list == val" ; x is a matrix of the same size as list that has 1 where the element is equal to val and a 0 evverywhere else
        % what exactly does it mean for a labelmatrix to "equal" an integer???
        particle = label_0 == counter;

        % ind2sub takes linear indices and returns subscripts ...??
        % find(x) finds all the nonzero elements in a matrix x ..????

        % ok I figured it out: This returns an array of subscripts, and those subscripts point to every single non-zero value in particle
        % what does a non-zero value in particle mean? It means the that label_0 is equal to counter at that subscript.
        % This is necessary because find() returns an array of linear indices
        [xx, yy, zz] = ind2sub([t1,t2,t3], find(particle));

        disp([double(counter),double(max(label_0(:))),double(areas(counter)),double(max(areas))]);

        % so now we get the range of integer values in every direction where something lies at this height (height = "counter")
        xrange = min(xx):max(xx);  % 0 : x = { 0, 1, 2, ... , x-1, x }
        yrange = min(yy):max(yy);
        zrange = min(zz):max(zz);

        % if the currently selected area > .75 * the greatest area
        if areas(counter) > (0.75 * max(areas))
            % just a print statement
            disp([min(xx),max(xx),min(yy),max(yy),min(zz),max(zz)]);
        end

        % NEW VARIABLE particle_local
        % looks like this is taking a 3d partition of particle, or more descriptively, "cropping" particle to only contain non-zero values?
        particle_local = particle(xrange,yrange,zrange);
        clear particle;

        % "local" Euclidean Distance Map is a distance map of the inverted 3d partition of particle ??
        EDM_local = bwdist(~particle_local);
        % we now modify this iedm using our adaptive function
        EDM_local_1 = F_water_modifyedm_adaptive(-EDM_local,aratio);
        clear EDM_local;

        disp('watersheding');
        % using the stock watershed function here ??
        water_local = watershed(EDM_local_1);
        % put a 1 where particle_local == 1 and the watershed result == 0
        cutline = particle_local & (water_local == 0);

        ridge_1(xrange,yrange,zrange) = min(ridge_1(xrange,yrange,zrange),~cutline);

        if max(cutline(:)) == 0%remove this particle if it's not segmented this time
            label_0(label_0 == counter) = 0;
        end
        clear water_local;clear cutline;
    else
        label_0(label_0==counter)=0;
    end
end

if min(ridge_1)==1
    water.end=true;
end

if job.modi==2 %no iteration for the traditional method
    water.end=true;
end

water.ridge=water.ridge & ridge_1;
water.segment=water.segment & (label_0 > 0);%not useful after iteration
water.segment=water.segment & ridge_1;
