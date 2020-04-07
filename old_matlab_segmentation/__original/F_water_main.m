function handle=F_water_main(handle,job)


if job.cutlarge && min(handle.ridge(:))==1
    disp('first time!')
    aratio=max(job.aratio,job.aratio);
else
    aratio=job.aratio;
end


disp('labelling and calculating property')
temp_0=bwconncomp(handle.segment);
label_0=labelmatrix(temp_0);
label_0p=regionprops(temp_0,'area');
areas=[label_0p.Area];
clear temp_0;clear label_0p;

cri_size=job.minarea;
clear temp;

[t1,t2,t3]=size(handle.segment);
ridge_1=true(t1,t2,t3);

disp("Number of regions found:");
disp(max(label_0(:)));

for counter = 1:max(label_0(:))
    
    disp(counter);
    
    if (areas(counter)>=cri_size) 
        particle     =   label_0==counter;
        [xx, yy, zz] =   ind2sub([t1,t2,t3], find(particle));
        
        % disp([double(counter),double(max(label_0(:))),double(areas(counter)),double(max(areas))]);

        xd       = min(xx):max(xx);
        yd       = min(yy):max(yy);
        zd       = min(zz):max(zz);  
        
        if areas(counter)>0.75*max(areas)
            % disp([min(xx),max(xx),min(yy),max(yy),min(zz),max(zz)]);
        end
        
        particle_local=particle(xd,yd,zd);
        clear particle;
        
        EDM_local           = bwdist(particle_local);
        
        water_local=F_water_modifyedm_adaptive(-EDM_local,aratio);

        clear EDM_local;

        % messy ass code
        cutline=particle_local & (water_local==0);
        
        ridge_1(xd,yd,zd)=min(ridge_1(xd,yd,zd),~cutline);

        if max(cutline(:))==0%remove this particle if it's not segmented this time
            label_0(label_0==counter)=0;
        end

        clear water_local;clear cutline;
        
    else
        label_0(label_0==counter)=0;
    end
end

if min(ridge_1)==1
    handle.end=true;
end

if job.modi==2%no iteration for the traditional method
    handle.end=true;
end

handle.ridge=handle.ridge & ridge_1;
handle.segment=handle.segment &(label_0>0);%not useful after iteration
handle.segment=handle.segment & ridge_1;