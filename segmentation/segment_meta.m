%{  
    Antonio Leonti
    4.1.2020
    Adaptive bring-up segmentation function
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
        
        for j = 1 : calls
            fprintf("  ");
        end
        fprintf("region %4d / %4d \n", i, max(label, [], "all"));

        if stats.Volume(i) >= minVolume
            %% obtain the region of interest and create IEDM

            % find indices of data which contain the current region
            [x, y, z] = ind2sub(size(data), find(data == i));
            
            x = min(x):max(x);
            y = min(y):max(y);
            z = min(z):max(z); % ranges of indices in each dim

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
                    
                    for j = 1 : calls
                        fprintf("  ");
                    end
                    fprintf("going deeper...\n");
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
