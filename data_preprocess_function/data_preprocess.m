clear
clc
% ref:
% https://www.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html
% https://www.mathworks.com/help/textanalytics/ug/create-simple-text-model-for-classification.html
%data path
addpath('/data')
outpath = '/data_preprocess';

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
Y_cat = zeros(n_1 , 1);
for i = 1 : n_1
    if data_train(i).('requester_received_pizza')
        Y_cat(i) = 1;
    else
        Y_cat(i) = -1;
    end
end

train_txt = string(zeros(n_1 , 1));
for i = 1 : n_1
    train_txt(i , 1) = convertCharsToStrings(data_train(i).('request_text_edit_aware'));
end

test_txt = string(zeros(n_2 , 1));
for i = 1 : n_2
    test_txt(i , 1) = convertCharsToStrings(data_test(i).('request_text_edit_aware'));
end

% figure
% wordcloud(train_txt);
% title("Training Data")

giver = zeros(n_1,1);
for i = 1 : n_1
    if convertCharsToStrings(data_train(i).('giver_username_if_known')) ~= 'N/A'
        giver(i) = 1;
    end
end
idx_giver = find(giver == 1);

%%%%% Case1，刪除giver，刪除太短跟太長的單字，刪除出現頻率過少的單字 %%%%%
% giver是資料的一個featrue，代表的是用戶的ID
YTrain = Y_cat;
train_txt_train = train_txt;
test_txt_test = test_txt;

% 刪除giver
YTrain(idx_giver) = [];
train_txt_train(idx_giver) = [];

% 將text整理，刪掉不需要的字，及統一型態
% preprocessText_1 刪除太短跟太長
% preprocessText_2 保留太短跟太長
documents_train = preprocessText_1(train_txt_train);
documents_test = preprocessText_1(test_txt_test);

% 將整理好的text中的word編碼，寫成一個字典
bag_train = bagOfWords(documents_train);

% 刪除出現頻率過少單字
% Remove words from the bag-of-words model that do not
% appear more than two times in total
% Remove any documents containing no words from 
% the bag-of-words model, and remove the corresponding 
% entries in labels
bag_train = removeInfrequentWords(bag_train,2);
[bag_train,idx_removeEmpty] = removeEmptyDocuments(bag_train);
YTrain(idx_removeEmpty) = [];

% 去計算每一個text中出現的單字數，轉換成向量
XTrain = bag_train.Counts;
XTest = encode(bag_train,documents_test);
% 輸出轉換成數字的文本
save('XTrain_1.mat','XTrain')
save('YTrain_1.mat','YTrain')
save('XTest_1.mat','XTest')
clear XTrain YTrain XTest

%%%%% Case2，刪除giver，保留太短跟太長的單字，刪除出現頻率過少的單字 %%%%%
YTrain = Y_cat;
train_txt_train = train_txt;
test_txt_test = test_txt;

% 刪除giver
YTrain(idx_giver) = [];
train_txt_train(idx_giver) = [];

% 將text整理，刪掉不需要的字，及統一型態
% preprocessText_1 刪除太短跟太長
% preprocessText_2 保留太短跟太長
documents_train = preprocessText_2(train_txt_train);
documents_test = preprocessText_2(test_txt_test);

% 將整理好的text中的word編碼，寫成一個字典
bag_train = bagOfWords(documents_train);

% 刪除出現頻率過少單字
% Remove words from the bag-of-words model that do not
% appear more than two times in total
% Remove any documents containing no words from 
% the bag-of-words model, and remove the corresponding 
% entries in labels
bag_train = removeInfrequentWords(bag_train,2);
[bag_train,idx_removeEmpty] = removeEmptyDocuments(bag_train);
YTrain(idx_removeEmpty) = [];

% 去計算每一個text中出現的單字數，轉換成向量
XTrain = bag_train.Counts;
XTest = encode(bag_train,documents_test);
% 輸出轉換成數字的文本
save('XTrain_2.mat','XTrain')
save('YTrain_2.mat','YTrain')
save('XTest_2.mat','XTest')
clear XTrain YTrain XTest

%%%%% Case3，刪除giver，刪除太短跟太長的單字，保留出現頻率過少的單字 %%%%%
YTrain = Y_cat;
train_txt_train = train_txt;
test_txt_test = test_txt;

% 刪除giver
YTrain(idx_giver) = [];
train_txt_train(idx_giver) = [];

% 將text整理，刪掉不需要的字，及統一型態
% preprocessText_1 刪除太短跟太長
% preprocessText_2 保留太短跟太長
documents_train = preprocessText_1(train_txt_train);
documents_test = preprocessText_1(test_txt_test);

% 將整理好的text中的word編碼，寫成一個字典
bag_train = bagOfWords(documents_train);

% 刪除出現頻率過少單字
% Remove words from the bag-of-words model that do not
% appear more than two times in total
% Remove any documents containing no words from 
% the bag-of-words model, and remove the corresponding 
% entries in labels
% bag_train = removeInfrequentWords(bag_train,2);
% [bag_train,idx_removeEmpty] = removeEmptyDocuments(bag_train);
% YTrain(idx_removeEmpty) = [];

% 去計算每一個text中出現的單字數，轉換成向量
XTrain = bag_train.Counts;
XTest = encode(bag_train,documents_test);
% 輸出轉換成數字的文本
save('XTrain_3.mat','XTrain')
save('YTrain_3.mat','YTrain')
save('XTest_3.mat','XTest')
clear XTrain YTrain XTest

%%%%% Case4，刪除giver，保留太短跟太長的單字，保留出現頻率過少的單字 %%%%%
YTrain = Y_cat;
train_txt_train = train_txt;
test_txt_test = test_txt;

% 刪除giver
YTrain(idx_giver) = [];
train_txt_train(idx_giver) = [];

% 將text整理，刪掉不需要的字，及統一型態
% preprocessText_1 刪除太短跟太長
% preprocessText_2 保留太短跟太長
documents_train = preprocessText_2(train_txt_train);
documents_test = preprocessText_2(test_txt_test);

% 將整理好的text中的word編碼，寫成一個字典
bag_train = bagOfWords(documents_train);

% 刪除出現頻率過少單字
% Remove words from the bag-of-words model that do not
% appear more than two times in total
% Remove any documents containing no words from 
% the bag-of-words model, and remove the corresponding 
% entries in labels
% bag_train = removeInfrequentWords(bag_train,2);
% [bag_train,idx_removeEmpty] = removeEmptyDocuments(bag_train);
% YTrain(idx_removeEmpty) = [];

% 去計算每一個text中出現的單字數，轉換成向量
XTrain = bag_train.Counts;
XTest = encode(bag_train,documents_test);
% 輸出轉換成數字的文本
save('XTrain_4.mat','XTrain')
save('YTrain_4.mat','YTrain')
save('XTest_4.mat','XTest')
clear XTrain YTrain XTest