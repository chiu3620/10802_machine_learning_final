close all
clear
% ref:
% https://www.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html
% https://www.mathworks.com/help/textanalytics/ug/create-simple-text-model-for-classification.html
% clc
%data path
addpath('/data')

%%%%% data_preprocess %%%%%
% 讀取train data，我把train.json放在跟matlab同一個資料夾裡
% 如果不是放在同一個資料夾裡，可能需要再額外加入路徑名稱
filename = 'train_data.json';
filename_2 = 'test_data.json';
% 讀取完的檔案會存成struct
data_train = jsondecode(fileread(filename));
data_test = jsondecode(fileread(filename_2));
% 資料數量
n_1 = length(data_train);
n_2 = length(data_test);
% 建立YTrian
Category = string(zeros(n_1 , 1));
for i = 1 : n_1
    if data_train(i).('requester_received_pizza')
        Category(i) = 'sucess';
    else
        Category(i) = 'fail';
    end
end

Description = string(zeros(n_1 , 1));
for i = 1 : n_1
    Description(i , 1) = convertCharsToStrings(data_train(i).('request_text_edit_aware'));
end

test_txt = string(zeros(n_2 , 1));
for i = 1 : n_2
    test_txt(i , 1) = convertCharsToStrings(data_test(i).('request_text_edit_aware'));
end

data = table(Description,Category);
head(data)

%%%%% data visualization %%%%%

data.Category = categorical(data.Category);
% figure
% histogram(data.Category);
% % xlabel("Class")
% ylabel("Frequency")
% title("Class Distribution")

%The next step is to partition it into sets for training and validation. 
% Partition the data into a training partition and a held-out partition for validation and testing.
% Specify the holdout percentage to be 20%.
cvp = cvpartition(data.Category,'Holdout',0.2);
dataTrain = data(training(cvp),:);
dataValidation = data(test(cvp),:);

textDataTrain = dataTrain.Description;
textDataValidation = dataValidation.Description;
YTrain = dataTrain.Category;
YValidation = dataValidation.Category;

% figure
% wordcloud(textDataTrain);
% title("Training Data")

documentsTrain = preprocessText_deep(textDataTrain);
documentsValidation = preprocessText_deep(textDataValidation);
documentsTrain(1:5)
enc = wordEncoding(documentsTrain);
documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")

%%%%% creat deep learning model %%%%%

% 調整input參數
sequenceLength = 200;
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);
XValidation = doc2sequence(enc,documentsValidation,'Length',sequenceLength);

% 調整layer參數
inputSize = 1;
embeddingDimension = 256;
numHiddenUnits = 64;

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));

% 建構deep learning model
%     dropoutLayer(0.5)
layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

% 選擇迭代法
options = trainingOptions('sgdm', ...
    'Momentum',0.9,...
    'MiniBatchSize',16, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);

%train model
net = trainNetwork(XTrain,YTrain,layers,options);

% 使用模型進行預測
XPred = classify(net,XTrain);
acc = sum(XPred == YTrain)/numel(YTrain)
reportsNew = test_txt;
documentsNew = preprocessText_deep(reportsNew);
XNew = doc2sequence(enc,documentsNew,'Length',sequenceLength);
labelsNew = classify(net,XNew);

% 建立YTest
YTest = zeros(n_2 , 1);
for i = 1 : n_2
    if labelsNew(i) == 'sucess'
        YTest(i) = 1;
    else
        YTest(i) = 0;
    end
end
% writematrix(YTest,'YTest_text_50_32_64_drop.csv') 
