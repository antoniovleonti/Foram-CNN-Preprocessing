%{
Antonio Leonti
4.1.2020
Attempting to rewrite the code from the paper "Quantification of the
morphology of shelly carbonate sands using 3D images" by D. KONG and J.
FONSECA. This code is easier to read and better at its intended purpose.
%}

clear;

%% load &/ make dataset

load("data\partition.mat");


%% modify dataset (morphological operations etc.)

data = fill3d(data);


%% watershed

fprintf("Beginning segmentation...\n");

result = segment(data, 0.125, 26, 50000);

%% show &/ save results

cc = bwconncomp(result, 6);
lm = labelmatrix(cc);

rgb = label2rgb(lm(:,:,15),'jet','w','shuffle');

imshow(rgb);

save("results\partition", "data");