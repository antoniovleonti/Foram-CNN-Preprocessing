function filled = fill_xy(data)
    %UNTITLED5 Summary of this function goes here
    %   Detailed explanation goes here
    shape = size(data);
    
    for z = 1 : shape(3)
       data(1:end, 1:end, z) = imfill(data(1:end, 1:end, z),'holes');
    end
    
    filled = data;
end