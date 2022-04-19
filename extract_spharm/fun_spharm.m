%   -*- coding: utf-8 -*-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Spherical Harmonics (SPHARM) Decomposition
%   Given, radial and frequency sampling, perform forward 
%   SPHARM transform to get a 3D SPHARM representation of the volume.
%
%   Not for clinical use.
%   SPDX-FileCopyrightText: 2022 Medical Physics Unit, McGill University, Montreal, CAN
%   SPDX-FileCopyrightText: 2022 Thierry Lefebvre
%   SPDX-FileCopyrightText: 2022 Ozan Ciga
%   SPDX-FileCopyrightText: 2022 Peter Savadjiev
%   SPDX-License-Identifier: MIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function flmr_in = fun_spharm(pathimgnii,pathsegnii,Rmax,Lmax)

addpath('\spharm\nfft-3.5.0-mexw64-openmp\nfsft\')
fprintf('Number of threads: %d\n', nfsft_get_num_threads());
addpath('\functionDep')
myfilepath = pathimgnii;
myfilepathsegmentations = pathsegnii;

%% Load images and segmentations
imaVOL = niftiread(convertCharsToStrings(myfilepath));
entryseg = niftiread(convertCharsToStrings(myfilepathsegmentations));
testing = niftiinfo(convertCharsToStrings(myfilepath));
imgin = double(imaVOL);

IMG = zeros(size(imgin));

segm = [];
img = [];
test = [];
imverif = [];
segverif = [];

entryseg = double(entryseg);
entryseg1 = entryseg;
entryseg(entryseg~=0)=1;

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
            img(:,:,end) = imgin(:,:,i) .* test;
            imverif(:,:,end) = imgin(:,:,i);
            ab = ab+1;
        else
            test = [];
            test1=[];
            test = imfill((entryseg(:,:,i)));
            test1 = imfill((entryseg1(:,:,i)));
            segm(:,:,end+1) = double(test);
            segverif(:,:,end+1) = double(test1);
            img(:,:,end+1) = imgin(:,:,i) .* test;
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

[X,Y,Z] = meshgrid( 1 : xdim,  1 : ydim,  1 : zdim );

% x_ --> r_
[PH, TH, R] = cart2sph(X, Y, Z);

min_theta1 = min(TH(:));
max_theta1 = max(TH(:));

% Define max radius.
max_radius = max([ceil(norm([xdim/2])), ceil(norm([ydim/2])), ceil(norm([zdim/2]))]);

% Center X, Y, Z such that mean is at 0
X = X - median(X(:));
Y = Y - median(Y(:));
Z = Z - median(Z(:));

% x_ --> r_
[PH, TH, R] = cart2sph(X, Y, Z);

min_theta = min(TH(:));
max_theta = max(TH(:));

clear xdim ydim zdim X Y Z PH TH R

[xdim, ydim, zdim] = size(IMG);

% Account for anisotropic zdim
zdim = ceil(zdim*testing.PixelDimensions(3)/testing.PixelDimensions(1));
maxdim = max([xdim,ydim,zdim]);

rgrid = Rmax;

if maxdim<rgrid
    xdim =  xdim*2;
    ydim =  ydim*2;
    zdim =  zdim*2;
    maxdim= maxdim*2;
end

IMG = imresize3(IMG, [xdim,ydim,zdim]);

% Zero-pad to have a cube of equal dimensions in all directions
IMG = padarray(IMG,[floor((maxdim-xdim)/2) floor((maxdim-ydim)/2) floor((maxdim-zdim)/2)],0,'both');


%% Init Forward Spherical harmonics transform to coefficients.

% Interpolate image
F = my_gridinterp(mat2gray(IMG));

% Maximal angular degree(bandwidth) of Spherical Harmonics expansions
Lmax = Lmax;

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

%     rgrid = max_radius; % 2^6;
rs = linspace(0.5, max_radius, rgrid);
dr = rs(2)-rs(1);

% Init coefficients matrix from plan
fh = cell(1, rgrid); % SPHARM coefficient
fo = cell(1, rgrid);
fo0 = cell(1, rgrid);

% Fix angles. NFFT wants X to be theta&phi from [0-+2pi]
% but matlab uses [-pi-+pi] notation.
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
    
    % Adjoint discrete spherical Fourier transform (direct alg.)
    nfsft_adjoint(plan);
    
    % Get SPHARM coefficients in a matrix from plan
    fh{r_index} = f_hat(nfsft_get_f_hat(plan)); % Call to nfsftmex; Gateway routine to the NFSFT module
end

%% Radial Fourier Basis Functions
tic
disp(' Fwd & backwards FFT begins ');

% Depends on r (or rgrid)
flmr = zeros(2*Lmax+1, Lmax+1, rgrid);

center_idx = Lmax + 1;
rs2 = rs.^2;

flmn = zeros(1, rgrid);
almrk = zeros(Lmax + 1, 2*Lmax+1, 2*rgrid-1);

% For each r value, there is a 2Lmax+1 x Lmax+1 SPHARM coefficients matrix
fhd = cellfun(@(f) double(f),fh,'UniformOutput',false); % f_hat type doesn't work with matlab, convert cell elems to double first.

for test=1:length(fhd)
    itertest = fhd{test}.';
    flmr_in(:,:,test)=itertest./test;
end
toc

% Export final 3D matrix Lmax+1 x 2Lmax+1 x Rmax
flmr_in = double(abs(flmr_in));

end