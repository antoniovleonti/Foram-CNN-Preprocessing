% EDM already has negative values
function MEDM = F_water_modifyedm_adaptive(EDM, aratio)
% get the locations of all the minima (set minima to 1, everything else to 1)
LM = imregionalmin(EDM);
% get the actual values of minima (everything else is multiplied by 0)
% ".*" is element-wise multiplication
minima = EDM .* LM;

% we made it negative just to later get the absolute value??? anyway:
% get abs() of every value in minima, then make a sorted array of all unique values in that minima
values = unique(abs(minima));
% exclude the first value (the first value is the SMALLEST)
depths = values(2:end);
% complex conjugate transpose? ...do we have imaginary parts? (i dont think so)
depths = depths';

MEDM = EDM;

if numel(depths) >= 2 % at least two catchments found;

    temp1 = [depths(2:end),0]; % ??? why is a 0 added to the end?
    temp2 = temp1 - depths; % ???
    ind = find(temp2 == max(temp2));

    cri_depth = depths(ind(1));

    %fill all basins by f*H_0
    MEDM = imhmin(EDM,aratio * cri_depth);

    MEDM(MEDM < -cri_depth) = -cri_depth;

end
