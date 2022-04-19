%   -*- coding: utf-8 -*-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Turn segmented ROIs into Spherical ROIs
%   Batch run on all segmentations.
%
%   Not for clinical use.
%   SPDX-FileCopyrightText: 2022 Medical Physics Unit, McGill University, Montreal, CAN
%   SPDX-FileCopyrightText: 2022 Thierry Lefebvre
%   SPDX-FileCopyrightText: 2022 Peter Savadjiev
%   SPDX-License-Identifier: MIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
warning off;

myfilepath = 'MYPROJECTFILEPATH/IMG';
myfilepathsegmentations = 'MYPROJECTFILEPATH/SEG';
myfilepathbase = 'MYPROJECTFILEPATH';
nyfilepathsave = 'MYPROJECTFILEPATH/SPHERESEG';

listdir = dir(myfilepath);
listdir(1) = [];
listdir(1) = [];
[listSize, J] = size(listdir);

listdirsegmentations = dir(myfilepathsegmentations);
listdirsegmentations(1) = [];
listdirsegmentations(1) = [];

for iii = 1:listSize
    %% Volume in spherical coordinates centered at r_ = 0_

    entryseg = niftiread([myfilepathsegmentations, sprintf('%s',listdirsegmentations(iii).name)]); 
    testing = load_nii([myfilepath, sprintf('%s',listdir(iii).name)]);

    disp(listdir(iii).name)
    disp(listdirsegmentations(iii).name)
    
    segmout = double(entryseg);
    segmout(segmout==2)=1;
    
    regions = regionprops3(segmout,"Centroid","PrincipalAxisLength");
    
    [xdim, ydim, zdim] = size(segmout);
    [X,Y,Z] = meshgrid( 1 : ydim,  1 : xdim,  1 : zdim );
    
    %% Turn into a sphere
    segmout(((X-regions.Centroid(1)).^2+(Y-regions.Centroid(2)).^2+((Z-regions.Centroid(3)).*ceil(testing.hdr.hist.srow_z(3)/testing.hdr.hist.srow_x(1))).^2)<(max(regions.PrincipalAxisLength/1.5)^2))=1;

    
    if ~exist(myfilepathsave, 'dir')
        mkdir(myfilepathsave)
    end

    
    nii1 = make_nii(segmout, [testing.hdr.hist.srow_x(1) testing.hdr.hist.srow_x(1) testing.hdr.hist.srow_z(3)], testing.hdr.hist.originator(1:3));
    save_nii(nii1, [myfilepathsave, listdir(iii).name]);

    clear segmout nii1 entryseg testing X Y Z
    
    
end

