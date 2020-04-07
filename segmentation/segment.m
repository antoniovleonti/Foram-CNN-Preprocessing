%{  
    Antonio Leonti
    4.1.2020
    Adaptive bring-up segmentation function
%}

function result = segment(data, ratio, conn, minVolume)

    result = segment_meta(data, ratio, conn, minVolume, 0);
end
