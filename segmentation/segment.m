%{  
Antonio Leonti
4.1.2020
This is the driver function for segment_meta(). The purpose of this
function is simply to make it so the user doesn't need to specify the state
parameter "calls," which is used for formatting console output.
%}

function result = segment(data, ratio, conn, minVolume)

    result = segment_meta(data, ratio, conn, minVolume, 0);
end
