clear
clc
% data path
addpath('/data_preprocess');

%%%%%%%%%%%%%%% KNN_1
k = 1;

%%%%%% Data 1 %%%%%%
load('XTrain_1.mat');
load('YTrain_1.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_1.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-1-case1, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);

%%%%%% Data 2 %%%%%%
load('XTrain_2.mat');
load('YTrain_2.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_2.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-1-case2, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);

%%%%%% Data 3 %%%%%%
load('XTrain_3.mat');
load('YTrain_3.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_3.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-1-case3, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);

%%%%%% Data 4 %%%%%%
load('XTrain_4.mat');
load('YTrain_4.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_4.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-1-case4, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);

%%%%%%%%%%%%%%% KNN_5
k = 5;

%%%%%% Data 1 %%%%%%
load('XTrain_1.mat');
load('YTrain_1.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_1.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-5-case1, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);

%%%%%% Data 2 %%%%%%
load('XTrain_2.mat');
load('YTrain_2.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_2.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-5-case2, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);

%%%%%% Data 3 %%%%%%
load('XTrain_3.mat');
load('YTrain_3.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_3.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-5-case3, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);

%%%%%% Data 4 %%%%%%
load('XTrain_4.mat');
load('YTrain_4.mat');
XTrain = full(XTrain);
rng(1); % For reproducibility
model = fitcknn(XTrain,YTrain,'NumNeighbors',k,'Standardize',1);
save('model_4.mat','model');
fprintf('model done\n');
XPred = predict(model,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain);
fprintf('acc done\n');
model = crossval(model,'KFold',5);
VError = kfoldLoss(model);
fprintf('knn-5-case4, accuracy = %1.5f, 5-fold cross-validation = %1.5f\n', acc, VError);
