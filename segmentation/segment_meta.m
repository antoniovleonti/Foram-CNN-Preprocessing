%{  
Antonio Leonti
4.1.2020
This is a reimplmentation of the algo from the paper "Quantification of the
morphology of shelly carbonate sands using 3D images" by D. KONG and J.
FONSECA (http://dx.doi.org/10.1680/jgeot.16.P.278). This code is easier to
read and faster than the original code. It's also recursive, which is cool.
%}

function result = segment_meta(data, ratio, conn, minVolume, calls)
    %% identify disjoint regions in data

    % get connected components
    cc = bwconncomp(data, conn);
    % get volume for each connected component
    stats = regionprops3(cc, "Volume");
    % create a label matrix of data from connected components
    label = labelmatrix(cc);

    %% for each of the regions...

    for i = 1 : max(label, [], "all")
        
        fprintf("%*c %d / %d \n", 2*calls, '|', i, max(label, [], "all"));

        if stats.Volume(i) >= minVolume
            %% obtain the region of interest and create IEDM

            % find indices of data which contain the current region
            [x, y, z] = ind2sub(size(data), find(data == i));
            x = min(x):max(x); % ranges of indices in each dim
            y = min(y):max(y);
            z = min(z):max(z); 
            % crop data and only include this region's elements
            crop = data(x, y, z) == i;
            
            % now create distance map
            iedm = -bwdist(~crop);

            %% modify local IEDM (if needed)

            % make list of all minima (do not include 0)
            tmp = unique(abs(imregionalmin(iedm) .* iedm));
            depths = tmp(2:end)'; % drop the zero
            
            if numel(depths) >= 2
                %% perform bring-up modification & segmentation
                
                % get index i of greatest difference depth_i - depth_i+1
                [~, index] = max([depths(2:end), NaN] - depths);
                % get d_i
                H = depths(index);
                
                % fill all the basins by H (imhmin)
                iedm = imhmin(iedm, H * ratio);
                % clip extremely deep basins to –H(1 – ratio)
                iedm(iedm < -H) = -H;
                
                % perform watershed segmentation
                segmented = watershed(iedm);
                
                %% repeat process (go deeper) iff new regions were found

                % determine if region was cut
                if any(crop & (segmented == 0), 'all')
                    
                    segmented = segment_meta(segmented, ratio, conn, minVolume, calls+1);
                end
                
                % transfer cuts to data
                data(x, y, z) = data(x, y, z) & ~(crop & (segmented == 0));
                
            end
        end
    end
    
    % return the data
    result = data;
end