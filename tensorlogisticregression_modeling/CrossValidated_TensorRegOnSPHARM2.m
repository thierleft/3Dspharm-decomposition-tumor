%   -*- coding: utf-8 -*-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Cross-validated TensorReg
%   Run on the training dataset to find optimal TensorReg matrix for 
%   classifying SPHARM coefficients 
%   Optional: TensorRegOnSPHARM3.m can be ran directly too.
%
%   SPDX-FileCopyrightText: 2022 Medical Physics Unit, McGill University, Montreal, CAN
%   SPDX-FileCopyrightText: 2022 Thierry Lefebvre
%   SPDX-FileCopyrightText: 2022 Peter Savadjiev
%   SPDX-License-Identifier: MIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all;
clear all;

% Uses https://hua-zhou.github.io/TensorReg/ package

showFig = 0; % Set to 1 to see all figures
range1 = 50; % Number of repeats

AUCTrainList = [];
AUCValList = [];
BList = cell(1,range1);

for bootidx = 1:range1 
    coeffList = [];
    labelList = [];
    aList = [];
    myfilepathbase = 'SAVE_TRAIN/';
    
    % Assuming coefficients are separated in two folders of outcomes (0 vs 1)
    for iii = 1:2
        
        % Get list of all subfolders.
        listdirSubFolder = dir(myfilepathbase);
        listdirSubFolder(1) = [];
        listdirSubFolder(1) = [];
        
        for idx=1:length(listdirSubFolder)
            aList = [aList, string(listdirSubFolder(idx).name)];
        end
        
        for k=1:length(listdirSubFolder)
            
            itFile = open([allSubFolders, listdirSubFolder(k).name]);
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
    
    
    %% Randomly sample data for CV  
    % Cross validation (train: 75%, validation: 25%)
    cv = cvpartition(size(coeffList,3),'HoldOut',0.25);
    idx = cv.test;
    
    % Separate to training and validation data
    dataTrain = coeffList(:,:,~idx);
    dataTest  = coeffList(:,:,idx);
    
    labelListTrain = labelList(~idx);
    labelListTest = labelList(idx);
    
    %% TensorReg 
    M = tensor(dataTrain);
    p0 = 1;
    b0 =  2.756; % random start
    
    n = size(dataTrain,3);  % sample size
    X = ones(n,p0);         % n-by-p regular design matrix
    y = labelListTrain';    % Labels

    % Lambdas to test   
    gridpts = 100;
    lambdas = zeros(1,gridpts);
    gs = 2/(1+sqrt(5));
    B = cell(1,gridpts);
    AIC = zeros(1,gridpts);
    BIC = zeros(1,gridpts);
    maxlambda = 100;
    
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
    if showFig==1
        %% Figures
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
    end
    BIC(BIC==min(BIC)) = Inf; % Take the second 
    [mini, minIdx] = min(BIC);
    
    %% Classification
    classprob = [];  
    realB=double(B{gridpts-minIdx});
    
    BList{bootidx}=realB;
    iter = 1;
    
    for alpha= 1:n
        testM = double(M(:,:,alpha));
        BX = sum(testM(:).*realB(:));
        linear_predictor = beta0+BX;
        probability_of_being_a_1 = exp(linear_predictor) / (1 + exp(linear_predictor));
        classprob = [classprob probability_of_being_a_1];      
        iter = iter+1;
    end

    
    %% Validating
     
    coeffListVal = dataTest;
    labelListVal = labelListTest;
    aListVal = [];
    
    MVal = tensor(coeffListVal);
    classprobVal=[];
    nVal = size(coeffListVal,3);
    
    for alpha= 1 :nVal
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

    % AUC Train
    [~,~,~,AUCTrain,~,~]=perfcurve(labelListTrain,classprob,1);
    AUCTrainList = [AUCTrainList AUCTrain];

    
    % AUC Val
    [~,~,~,AUCVal,~,~]=perfcurve(labelListVal,classprobVal,1);
    AUCValList = [AUCValList AUCVal];
        
end

figure;
plot(linspace(1,range1,range1), AUCTrainList,'o'); hold on;
plot(linspace(1,range1,range1), AUCValList,'x');
legend('Training', 'Validation')
xlabel('Bootstrap index')
ylabel('AUC')

% Optimal TensorReg is selected based on performance consistency between
% training and validation splits
diffAUC = AUCTrainList - AUCValList - AUCValList;
[minAUC, AUCidx] = min(diffAUC);

OptimalB = BList{AUCidx}; % Other selection metrics could be implemented here

% Plot optimal TensorReg matrix
figure;
imagesc(OptimalB);
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

% Assess AUC on the whole training dataset
classprob = [];
realB = OptimalB;
M = tensor(coeffList);
iter = 1;
classprob = [];
for alpha= 1:size(labelList,2)
    testM = double(M(:,:,alpha));
    BX = sum(testM(:).*realB(:));
    linear_predictor = beta0+BX;
    probability_of_being_a_1 = exp(linear_predictor) / (1 + exp(linear_predictor));
    classprob = [classprob probability_of_being_a_1];
    iter = iter+1;
end

% AUC of the cross-validated TensorReg classifier on the whole training dataset
[~,~,~,AUCTrainTotal,~,~] = perfcurve(labelList,classprob,1)

% Save optimal TensorReg classifier
save('TensorRegClassifier.mat','OptimalB');
