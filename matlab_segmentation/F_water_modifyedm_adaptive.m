% EDM already has negative values
function MEDM=F_water_modifyedm_adaptive(EDM,aratio)
% get the locations of all the minima (set minima to 1, everything else to 1)
LM = imregionalmin(EDM);
% get the actual values of minima (everything else is multiplied by 0)
minima = EDM .* LM;

values = unique(abs(minima));
depths = values(2:end);
depths = depths';
basins = numel(depths);

MEDM = EDM;

if basins >= 2%at least two catchments found;
    temp1 = [depths(2:end),0];
    temp2 = temp1 - depths;
    ind = find(temp2 == max(temp2));

    cri_depth = depths(ind(1));

    %fill all basins by f*H_0
    MEDM = imhmin(EDM,aratio * cri_depth);
    MEDM(MEDM <- cri_depth) =- cri_depth;

end
