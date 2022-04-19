%   -*- coding: utf-8 -*-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Spherical Harmonics (SPHARM) Decomposition AND Reconstruction.
%   Given, radial and frequency sampling, perform forward and backward
%   SPHARM transform to get the reconstructed volume of the original image.
%   The volumetric image is saved as .nii.
%
%   Not for clinical use.
%   SPDX-FileCopyrightText: 2022 Medical Physics Unit, McGill University, Montreal, CAN
%   SPDX-FileCopyrightText: 2022 Thierry Lefebvre
%   SPDX-FileCopyrightText: 2022 Ozan Ciga
%   SPDX-FileCopyrightText: 2022 Peter Savadjiev
%   SPDX-License-Identifier: MIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
warning off;

% Maximal radial expansion 
rgrid = 25;

% Maximal angular degree(bandwidth) of Spherical Harmonics expansions
Lmax = 25;

% Link to compilation of https://www-user.tu-chemnitz.de/~potts/nfft/nfsft.php
% Have to be adjusted and compiled before running this function
addpath('nfft-3.5.2-matlab-openmp/nfsft') % Adjust path

fprintf('Number of threads: %d\n', nfsft_get_num_threads());

myfilepath = 'MYPROJECTFILEPATH/IMG/';
myfilepathsegmentations = 'MYPROJECTFILEPATH/SEG/';
myfilepathbase = 'MYPROJECTFILEPATH/';
myfilepathsave = 'MYPROJECTFILEPATH/RECON_IMG';

listdir = dir(myfilepath);
listdir(1) = [];
listdir(1) = [];
[listSize, J] = size(listdir);

listdirsegmentations = dir(myfilepathsegmentations);
listdirsegmentations(1) = [];
listdirsegmentations(1) = [];


for iii = 1:size(listdir,2)

    % Get file path and load images
    imaVOL = niftiread([myfilepath, sprintf('%s',listdir(iii).name)]); 
    entryseg = niftiread([myfilepathsegmentations, sprintf('%s',listdirsegmentations(iii).name)]); 
    testing = load_nii([myfilepath, sprintf('%s',listdir(iii).name)]);
    
    disp(listdir(iii).name)
    disp(listdirsegmentations(iii).name)
    imgin = double(imaVOL);
    
    IMG = zeros(size(imgin));

    segm = [];
    img = [];
    test = [];
    imverif = [];
    segverif = [];
    
    %% Convert spherical volume to shape of segmentation for reconstruction of shape and texture
    entryseg = double(entryseg);
    entryseg1 = entryseg;
    entryseg(entryseg==2)=1;
    
    ab=1;
    for i = 1:size(entryseg,3)
        if sum(sum(entryseg(:,:,i))) ~= 0
            if ab==1
                test = [];
                test1= [];
                test1 = imfill((entryseg1(:,:,i)));
                test = imfill((entryseg(:,:,i)));
                segm(:,:,end) = double(test);
                segverif(:,:,end) = double(test1);
                img(:,:,end) = imgin(:,:,i) .* double(test);
                imverif(:,:,end) = imgin(:,:,i);
                ab = ab+1;
            else
                test = [];
                test1=[];
                test = imfill((entryseg(:,:,i)));
                test1 = imfill((entryseg1(:,:,i)));
                segm(:,:,end+1) = double(test);
                segverif(:,:,end+1) = double(test1);
                img(:,:,end+1) = imgin(:,:,i) .* double(test);
                imverif(:,:,end+1) = imgin(:,:,i);
                
            end
        end
    end
    
    IMG = img;
    
    ab = 1;
    for aa = 1:size(IMG, 2)
        if  ~sum(sum(IMG(:,ab,:)))
            IMG(:,ab,:) = [];
        else
            ab = ab+1;
        end
    end
    
    ab = 1;
    for aaa = 1:size(IMG, 1)
        if  ~sum(sum(IMG(ab,:,:)))
            IMG(ab,:,:) = [];
        else
            ab = ab+1;
        end
    end
    
    ab = 1;
    for aaa = 1:size(IMG, 3)
        if  ~sum(sum(IMG(:,:,ab)))
            IMG(:,:,ab) = [];
        else
            ab = ab+1;
        end
    end
    

    [xdim, ydim, zdim] = size(IMG);
    zdim = floor(zdim*testing.hdr.hist.srow_z(3)/testing.hdr.hist.srow_x(1));
    maxdim = max([xdim,ydim,zdim]);
    
    IMG = imresize3(IMG, [xdim,ydim,zdim],'method','nearest');
    IMG = padarray(IMG,[floor((maxdim-xdim)/2) floor((maxdim-ydim)/2) floor((maxdim-zdim)/2)],0,'both');
    
    % Define max radius.
    max_radius = max([ceil(norm([xdim/2])), ceil(norm([ydim/2])), ceil(norm([zdim/2]))]);
    
    
    [X,Y,Z] = meshgrid( 1 : xdim,  1 : ydim,  1 : zdim );
    
    % x_ --> r_
    [PH, TH, R] = cart2sph(X, Y, Z);
    
    min_theta1 = min(TH(:));
    max_theta1 = max(TH(:));
    
    
    % Center X, Y, Z such that mean is at 0
    X = X - median(X(:));
    Y = Y - median(Y(:));
    Z = Z - median(Z(:));
    
    % x_ --> r_
    [PH, TH, R] = cart2sph(X, Y, Z);
    
    min_theta = min(TH(:));
    max_theta = max(TH(:));
    
    clear xdim ydim zdim X Y Z PH TH R
    close all 
    
    %% Init Forward Spherical harmonics transform to coefficients.
    
    % Interpolate image
    F = my_gridinterp( IMG );
    
    % threshold (affects accuracy, 1000 is the default)
    kappa = 1000;
    
    % precomputation
    nfsft_precompute(Lmax,kappa);
    
    % Gauss-Legendre interpolatory quadrature nodes for L. See gl_oz.m
    %   X is X = GL(N) generates the (2N+2)*(N+1) Gauss-Legendre nodes and returns a
    %   2x[(2N+2)*(N+1)] matrix X containing their spherical coordinates. The first
    %   row contains the longitudes in [0,2pi] and the second row the colatitudes in
    %   [0,pi].
    %   [X,W] = GL(N) in addition generates the quadrature weights W. The resulting
    %   quadrature rule is exact up to polynomial degree 2*N.
    [X,W] = gl_oz(Lmax, min_theta, max_theta); % X == [theta; phi]
    
    % number of nodes
    M = size(X,2);
    
    % Create plan.
    % Advanced plan initialization routine
    plan = nfsft_init_advanced(Lmax,M,NFSFT_NORMALIZED);
    
    % Set nodes in plan
    nfsft_set_x(plan,X);
    
    % Node-dependent precomputation (for NFFT)
    nfsft_precompute_x(plan);
    
    rs = linspace(0.5, max_radius, rgrid);
    dr = rs(2)-rs(1);
    
    % Init coefficients matrix from plan
    fh = cell(1, rgrid); % SPHARM coefficient
    fo = cell(1, rgrid);
    fo0 = cell(1, rgrid);
    
    % Fix angles. NFFT wants X to be theta&phi from [0-+2pi]
    % but matlab uses [-pi-+pi] notation
    idx = find(X(1,:)>pi);
    X(1,idx) = X(1,idx) - 2*pi;
    X(2,:) = pi/2 - X(2,:);
    
    %% Forward Spherical harmonics transform to coefficients
    % f(r_) --> flmn
    % Note: Radial and angular components evaluated separately
    disp(' Fwd SPHARM begins ');
    for r_index= 1 : rgrid
        [sx,sy,sz] = sph2cart(X(1,:),X(2,:),rs(r_index));
        
        % Set function values in plan *considering weights and radius rs
        nfsft_set_f(plan, (rs(r_index)*F(sx, sy, sz))'.*W' );
        
        % Adjoint discrete spherical Fourier transform 
        nfsft_adjoint(plan);
        
        % Get SPHARM coefficients in a matrix from plan
        fh{r_index} = f_hat(nfsft_get_f_hat(plan)); % Call to nfsftmex; Gateway routine to the NFSFT module
    end
    
    %% Radial Fourier Basis Functions
    disp(' Fwd & backwards FFT begins ');
    
    % Depends on r (or rgrid)
    flmr = zeros(2*Lmax+1, Lmax+1, rgrid);
    
    center_idx = Lmax + 1;
    rs2 = rs.^2;
    
    flmn = zeros(1, rgrid);
    almrk = zeros(Lmax + 1, 2*Lmax+1, 2*rgrid-1);
    
    % For each r value, there is a 2Lmax+1 x Lmax+1 SPHARM coefficients matrix
        fhd = cellfun(@(f) double(f),fh,'UniformOutput',false); 
    for L = (0 : Lmax)
        for m = 0 : L
            % f_{lm}(r)
            flmr_tmp = cellfun(@(c)c(center_idx+m, L + 1), fhd);
            flmn =  dr * fft(flmr_tmp) * exp(-L*(L+1)*0.00001) * (exp(length(flmr_tmp)/max_radius)*1/sqrt(max_radius));
            
            if m>0
                
                almrk(L+1, L+1-m, 1:rgrid-1) = (-1)^m*conj(flmn(end:-1:2));
                
            end
            
            almrk(L+1, L+1+m, rgrid:end) = flmn;
            
            % Reconstr.
            res_pos = 1./dr * ifft(almrk(L+1, L+1+m, rgrid:end)) * (exp(length(flmr_tmp)/max_radius)*1/sqrt(max_radius))*length(flmr_tmp);
            res_neg = 1./dr * fft(almrk(L+1, L+1+m, rgrid:-1:1)) * (exp(length(flmr_tmp)/max_radius)*1/sqrt(max_radius));
            res = squeeze(res_pos + res_neg)- almrk(L+1, L+1+m, rgrid)*1/sqrt(max_radius);
            flmr(center_idx + (m), L + 1, :) = flmr_tmp;res;

            if m>0 flmr(center_idx + (-m), L + 1, :) = conj(flmr_tmp); end
        end
    end
    
    for test=1:length(fhd)
        itertest = fhd{test}.';
        flmr_in(:,:,test)=itertest./test;
    end
    

    Bfin = permute(realB, [3 1 2]);
    for kkk = 1:size(flmr,1)
        flmr(kkk,:,:) = flmr(kkk,:,:).*Bfin;
    end   

    %% Backward Spherical harmonics transform
    disp(' Backwards SPHARM begins ');
    for r_index= 1 : size(flmr, 3)
        nfsft_set_f_hat(plan,double(1./rs(r_index)*flmr(:, :, r_index)));
        nfsft_trafo(plan);
        fo{r_index} = nfsft_get_f(plan)';
    end
    disp(' Backwards SPHARM ends ');

    % Reconstruct image from spherical points
    disp(' Reconstruction begins ');

    sxt = []; syt = []; szt = []; 
    fo_array = []; 
    
    for i = 1 : numel(rs)
        [i,numel(sxt)];
        [sx,sy,sz] = sph2cart( X(1,:), X(2,:), rs(i) );
        sxt = [sxt; sx']; 
        syt = [syt; sy']; 
        szt = [szt; sz']; 
        fo_array = [fo_array; fo{i}' ]; 
    end
    sx = sxt; 
    sy = syt; 
    sz = szt; 
    clear sxt syt szt
    
    
    % f0 is the reconstructed image after backward spharm transform
    F_sct = scatteredInterpolant(sx,sy,sz,reshape(real(fo_array),[],1),'linear','none');
    
    [meshx, meshy, meshz] = meshgrid( (1:size(IMG,2)) , ...
        (1:size(IMG,1)) , ...
        1:size(IMG,3) );
    meshx = meshx - median(meshx(:));
    meshy = meshy - median(meshy(:));
    meshz = meshz - median(meshz(:));
    
    % Reconstructed image
    myimg = F_sct(meshx,meshy,meshz);
    
    % Multi-scale similarity index between original and SPHARM-recon images
    [score1,map1] = multissim3(myimg,IMG); 
    
    % Save as nii
    nii2 = make_nii(myimg, [1 1 1], [0 0 0]);
    save_nii(nii2, [myfilepathsave,'/',listdir(iii).name]);

end

