# 10802_machine_learning_final
* 課程名稱：機械學習
* 授課老師：李育杰教授
* 修課最終成績：A+
* 使用資料：Random Acts of Pizza Predicting altruism through free pizza
* 資料來源：https://www.kaggle.com/c/random%20acts%20of%20pizza/
主題簡介：
　在Reddit平台上曾舉辦過一個請求免費披薩的活動，參加者透過撰寫請求披薩的貼文，以此獲得他人的目光，而得到免費的披薩。我們要透過 Reddit平台提供的文本，以及這些文本最終有沒有得到披薩的資訊，以此建立一個模型，預測甚麼樣的文本可以獲得免費的披薩。
* 檔案說明：
　一共有兩個資料夾，第一個data_preprocess_function是資料前處理，第二個model_run是模型的建立與預測。
  * 資料前處理的做法主要是，先做斷詞，接著停用詞(stopword)刪除，再最後一步是將單字都還原成原型，像是：將過去時態的動詞變成現在時態；複數的名詞變成單數。再來處理太長、太短的句子，以及出現頻率過低的單字，最終使用bag of words，將文字變為向量。
  * 模型的建立與預測的部分，我們一共使用SVM, SSVM, KNN, LSTM，分別訓練出4種模型。其中SSVM是使用李育杰個人網站上提供的程式碼，其餘3個是使用matlab內建的函數。
* ref:
  * 李育杰。Smooth Support Vector Machine Toolbox。Retrieved May 19, 2021。https://dsmilab.github.io/Yuh-Jye-Lee/
  * Matlab。Classify Text Data Using Deep Learning。Retrieved May 19, 2021, from https://www.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html
  * Matlab。Create Simple Text Model for Classification。Retrieved May 19, 2021, from https://www.mathworks.com/help/textanalytics/ug/create-simple-text-model-for-classification.html
