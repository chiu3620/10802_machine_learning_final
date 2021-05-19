clear
clc
% data path
addpath('/data_preprocess');

%%%%%% Data 1 %%%%%%
load('XTrain_1.mat');
load('YTrain_1.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcsvm(XTrain, YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
% save('model_1.mat','model');
fprintf('model done');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('svm-case1, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);
clear XTrain YTrain acc VError model

%%%%%% Data 2 %%%%%%
load('XTrain_2.mat');
load('YTrain_2.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcsvm(XTrain, YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
% save('model_2.mat','model');
fprintf('model done');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('svm-case2, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);
clear XTrain YTrain acc VError model

%%%%%% Data 3 %%%%%%
load('XTrain_3.mat');
load('YTrain_3.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcsvm(XTrain, YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
% save('model_3.mat','model');
fprintf('model done');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('svm-case3, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);
clear XTrain YTrain acc VError model

%%%%%% Data 4 %%%%%%
load('XTrain_4.mat');
load('YTrain_4.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcsvm(XTrain, YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
% save('model_4.mat','model');
fprintf('model done');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('svm-case4, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);
clear XTrain YTrain acc VError model