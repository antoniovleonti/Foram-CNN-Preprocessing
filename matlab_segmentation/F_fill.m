function filled=F_fill(dark,piece,enhanced)
%disp('fill the image');
dark1=dark;



if piece==3
    disp('filling in three directions');
    
    dark2=permute(dark,[3 1 2]);
    dark3=permute(dark,[2 3 1]);
    
    
    
    filled1=F_fillpiecebypiece(dark1,enhanced);
    %disp('fill in row');
    filled2_t=F_fillpiecebypiece(dark2,enhanced);
    %disp('fill in col');
    filled3_t=F_fillpiecebypiece(dark3,enhanced);
    filled2=permute(filled2_t,[2 3 1]);
    filled3=permute(filled3_t,[3 1 2]);

    temp=(~filled1) & (~filled2) & (~filled3);
    filled=~temp;
elseif piece==1
    disp('filling vertical pieces');
    filled=F_fillpiecebypiece(dark1,enhanced);
end




