function MEDM=F_water_modifyedm_old(EDM,job)%EDM already has negative values

[row,col,lev]=size(EDM);
di=ndims(EDM);
LM          =   imregionalmin(EDM);
Minvalues   =   EDM.*LM;

values=unique(Minvalues);
v1=values(1);%find exmtreme value;

if numel(values)<=2
    v2=0;%values(end);
else
    v2=min(median(values(2:end-1)),mean(values(2:end-1)));%find second value;
end

ratio=abs(v2)/abs(v1);
MEDM=EDM;

if v2==0 || ratio>=job.diff%using imhmin
    %disp('narrowly ranged size');
    filldepth       = abs(job.uratio*v1);
    MEDM            = imhmin(EDM,filldepth);
 
else
    disp('widely ranged size');
    temp=bwconncomp(LM);
    L=labelmatrix(temp);
    for mm=1:temp.NumObjects
        MP=L==mm;
        [ii,jj,kk]=ind2sub(size(MP),find(MP));
        x0=mean(ii);y0=mean(jj);z0=mean(kk);
        x=floor(x0);y=floor(y0);z=floor(z0);
        
        Mvalue=Minvalues(x,y,z);
        R=abs(Mvalue*job.dratio);
        range=round(R)+1;

        ib=max(1,x-range);ie=min(row,x+range);
        jb=max(1,y-range);je=min(col,y+range);
        
        if di==3
            kb=max(1,z-range);ke=min(lev,z+range);
        else
            kb=1;ke=1;
        end
        for i=ib:ie
            for j=jb:je
                for k=kb:ke
                    if (i-x0)^2+(j-y0)^2+(k-z0)^2<=R^2
                        MEDM(i,j,k)=min(Mvalue,MEDM(i,j,k));
                    end
                end
            end
        end
    end
end
