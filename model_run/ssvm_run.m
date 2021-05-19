clear
clc
% ssvm function path
addpath('/ssvm')
% data path
addpath('/data_preprocess');

%%%%%% Data 1 %%%%%%
load('XTrain_1.mat');
load('YTrain_1.mat');
Result = hibiscus(YTrain, XTrain, '-s 0 -v 5 -r 1', '9-5');
save('result_1','Result')
clear XTrain YTrain Result

%%%%%% Data 2 %%%%%%
load('XTrain_2.mat');
load('YTrain_2.mat');
Result = hibiscus(YTrain, XTrain, '-s 0 -v 5 -r 1', '9-5');
save('result_2','Result')
clear XTrain YTrain Result

%%%%%% Data 3 %%%%%%
load('XTrain_3.mat');
load('YTrain_3.mat');
Result = hibiscus(YTrain, XTrain, '-s 0 -v 5 -r 1', '9-5');
save('result_3','Result')
clear XTrain YTrain Result

%%%%%% Data 4 %%%%%%
load('XTrain_4.mat');
load('YTrain_4.mat');
Result = hibiscus(YTrain, XTrain, '-s 0 -v 5 -r 1', '9-5');
save('result_4','Result')
clear XTrain YTrain Result