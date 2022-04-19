%   -*- coding: utf-8 -*-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Fit and apply TensorReg on the training and testing datasets for
%   classifying SPHARM coefficients 
%
%   SPDX-FileCopyrightText: 2022 Medical Physics Unit, McGill University, Montreal, CAN
%   SPDX-FileCopyrightText: 2022 Thierry Lefebvre
%   SPDX-FileCopyrightText: 2022 Peter Savadjiev
%   SPDX-License-Identifier: MIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;


% Uses https://hua-zhou.github.io/TensorReg/ package

coeffList = [];
labelList = [];
aList = [];

loadCV = 0; % 1 if cross-validated TensorReg matrix was produced
tensorRegCVpath = 'PATH_TO_TENSOR_REG_CV.../... .mat'; % If loadCV == 1

myfilepathbase = 'SAVE_TRAIN/'; % Should have outcome folders in SAVE (0 vs. 1)
myfilepathbaseVal = 'SAVE_TEST/';

for iii = 1:2 % Loop over the two outcome folders in SAVE
    
    %% Load all SPHARM matrices into a Tensor
    
    % Get list of all subfolders.
    listdirSubFolder = dir(myfilepathbase);
    listdirSubFolder(1) = [];
    listdirSubFolder(1) = [];
    
    for idx=1:length(listdirSubFolder)
        aList = [aList, string(listdirSubFolder(idx).name)];
    end
    
    for k=1:length(listdirSubFolder)
        
        itFile = open([myfilepathbase, listdirSubFolder(k).name]);
        itCoeff = itFile.flmr_in;
        itCoeff = squeeze((vecnorm(itCoeff,2,2))); % L-2 norm along the M dimension
        labelList = [labelList iii-1];
        
        if iii==1 && k == 1
            coeffList = itCoeff;
        else
            coeffList = cat(3,coeffList,itCoeff);
        end
        
    end
end

%% Fit TensorReg classifier on SPHARM matrices' tensor
M = tensor(coeffList);
p0 = 1;
b0 =  2.756; % Random

n = size(coeffList,3);    % sample size
X = ones(n,p0);   % n-by-p regular design matrix

y = labelList';

maxlambda = 100; % Max regularization
gridpts = 100;
lambdas = zeros(1,gridpts);
gs = 2/(1+sqrt(5));
B = cell(1,gridpts);
AIC = zeros(1,gridpts);
BIC = zeros(1,gridpts);

tic;
for i=1:gridpts
    if (i==1)
        B0 = [];
    else
        B0 = B{i-1};
    end
    lambda = maxlambda*gs^(i-1);
    lambdas(i) = lambda;
    [beta0,B{i},stats] = matrix_sparsereg(X,M,y,lambda,'binomial','B0',B0);
    AIC(i) = stats.AIC;
    BIC(i) = stats.BIC;
end
toc

figure; hold on;
set(gca,'FontSize',20);

ploti = 1;
for i=[gridpts round(gridpts/2) 1]
    ploti = ploti + 1;
    subplot(2,2,ploti);
    imagesc(-double(B{i}));
    colormap(gray);
    title({['nuclear norm,', ' \lambda=', ...
        num2str(lambdas(i))]; ['BIC=', num2str(BIC(i))]});
    axis square;
    axis tight;
end


figure;
set(gca,'FontSize',20);
semilogx(lambdas, AIC, '-+', lambdas, BIC, '-o');
xlabel('\lambda');
ylabel('BIC');
xlim([min(lambdas) max(lambdas)]);
title('Nuclear norm AIC/BIC');
legend('AIC', 'BIC', 'Location', 'northwest');
xlim([0.001 100])

BIC(BIC==min(BIC)) = Inf; % Take the second lambda minimizing BIC
[mini, minIdx] = min(BIC);

% TensorReg matrix fitted on all training dataset
realB=double(B{gridpts-minIdx}); % Regularization can also be selected based on produced figures

figure;
imagesc(realB);
colormap(jet);
colorbar();
caxis([-0.2 0.15])
axis square;
axis tight;
xlabel('Radius')
ylabel('Frequency')
ax = gca;
ax.FontSize = 14;
set(ax,'xtick', [0 5 10 15 20 25]);
set(ax,'ytick', [0 5 10 15 20 25]);

if loadCV == 1 % load cross-validated TensorReg classifier
    load(tensorRegCVpath)
    realB = OptimalB;
    
    figure;
    title('Cross-validated TensorReg Classifier')
    imagesc(realB);
    colormap(jet);
    colorbar();
    caxis([-0.2 0.15])
    axis square;
    axis tight;
    xlabel('Radius')
    ylabel('Frequency')
    ax = gca;
    ax.FontSize = 14;
    set(ax,'xtick', [0 5 10 15 20 25]);
    set(ax,'ytick', [0 5 10 15 20 25]);
end

%% Assess probabilities in training dataset

classprob = []; % Risk vector
for alpha= 1 :n
    
    testM = double(M(:,:,alpha));
    BX = sum(testM(:).*realB(:));    
    linear_predictor = beta0+BX;
    probability_of_being_a_1 = exp(linear_predictor) / (1 + exp(linear_predictor));
    classprob = [classprob probability_of_being_a_1];
    
    iter = iter+1;
end


%% Assess probabilities in testing dataset

coeffListVal = [];
labelListVal = [];
aListVal = [];

for iii = 1:2
    listdirSubFolder = dir(myfilepathbaseVal);
    listdirSubFolder(1) = [];
    listdirSubFolder(1) = [];
    for idx=1:length(listdirSubFolder)
        aListVal = [aListVal, string(listdirSubFolder(idx).name)];
    end
    
    for k=1:length(listdirSubFolder)
        
        itFile = open([myfilepathbaseVal, listdirSubFolder(k).name]);
        itCoeff = itFile.flmr_in;
        itCoeff = squeeze((vecnorm(itCoeff,2,2))); % L-2 norm along the M dimension
        labelListVal = [labelListVal iii-1];
        
        if iii==1 && k == 1
            coeffListVal = itCoeff;
        else
            coeffListVal = cat(3,coeffListVal,itCoeff);
        end
        
    end
end

MVal = tensor(coeffListVal);
classprobVal=[];
iter=1;
nVal = size(coeffListVal,3);

for alpha= 1:nVal
    testM = double(MVal(:,:,alpha));
    BX = sum(testM(:).*realB(:));
    linear_predictor = beta0+BX;
    probability_of_being_a_1 = exp(linear_predictor) / (1 + exp(linear_predictor));
    classprobVal = [classprobVal probability_of_being_a_1];
    
    iter = iter+1;
end

if mean(classprobVal(labelListVal==0))>mean(classprobVal(labelListVal==1))
    classprobVal = 1-classprobVal;
end

% Classification performance in training and testing dataset
[~,~,~,AUC_Train,~,~]=perfcurve(labelList,classprob,1,'NBoot',100);
[~,~,~,AUC_Test,~,~]=perfcurve(labelListVal,classprobVal,1,'NBoot',100);

%% Save risks in the training and testing datasets
save('predictions.mat','classprob','classprobVal','aList','aListVal','labelList','labelListVal','realB')


% For combining predictions from multiple image types, use scikit-learn
% implementation of logistic regression developed for radiomics pipeline to
% combine exported predictions for SPHARM decomposition of each image types