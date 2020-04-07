function bw_fill=F_fillpiecebypiece(bw,enhanced)
%should be 3D logical matrix
if nargin==1
    enhanced=0;
end

[row,col,level]=size(bw);

bw_fill=false(row,col,level);

for i=1:level
    %disp([i,level]);
    if enhanced
        bw_fill(:,:,i)=F_enhancedimfill2d(bw(:,:,i));
    else
        bw_fill(:,:,i)=imfill(bw(:,:,i),'holes');
    end
end
    